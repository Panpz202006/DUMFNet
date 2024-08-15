import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        return out


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class ABM(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in double original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        res = x.clone()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) + res

        return x


class MLP(nn.Module):
    def __init__(self, dim1, dim2, bias=False):
        super().__init__()

        self.qkv = nn.Linear(dim1, dim1 * 3, bias=bias)
        self.proj = nn.Linear(dim1 * 3, dim2)

    def forward(self, x):
        qkv = self.qkv(x)
        x = self.proj(qkv)

        return x


class MEM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dim1=input_dim
        self.dim2=output_dim
        self.CNN = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.dim2, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.dim2, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.MSA_attn = Attention(dim=self.dim1, num_heads=8)
        self.MSA_conv = nn.Conv2d(self.dim1, self.dim2, kernel_size=1)
        self.MLP = MLP(dim1=self.dim1, dim2=self.dim2)
        self.LSPVM_pvm = PVMLayer(input_dim=self.dim1, output_dim=self.dim2)
        self.ebn=nn.GroupNorm(4, self.dim2)
        self.p = 2

    def forward(self, x):
        b, c, h, w = x.shape
        window_mamba = x.clone()
        x_div = window_mamba.reshape(b, c, h // self.p, self.p, w // self.p, self.p).permute(0, 3, 5, 1, 2,
                                                                                             4).contiguous().view(
            b * self.p * self.p, c, h // self.p, w // self.p)

        mamba_out = self.LSPVM_pvm(x_div).reshape(b, self.dim2, h, w)
        cnn_out = self.CNN(x)
        attn_out = x.clone().view(b, h * w, -1)
        attn_out = self.MSA_attn(attn_out)
        attn_out = attn_out.view(b, self.dim1, h, w)
        attn_out = self.MSA_conv(attn_out)
        mlp_out = x.clone().view(b, h * w, -1)
        mlp_out = self.MLP(mlp_out)
        mlp_out = mlp_out.view(b, self.dim2, h, w)
        out = mamba_out + cnn_out + attn_out + mlp_out
        out = F.avg_pool2d(self.ebn(out), 2, 2)
        softmax_out = torch.softmax(out.clone(), dim=1)
        out = out * softmax_out
        out = F.gelu(out)
        return out


class DUMFNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.mem=MEM(input_dim=c_list[2], output_dim=c_list[3])
        self.encoder5 = PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        self.encoder6 = PVMLayer(input_dim=c_list[4], output_dim=c_list[5])

        if bridge:
            self.scab = ABM(c_list, split_att)

        self.decoder1 = PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        self.decoder2 = PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        self.decoder3 = PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.double_encoder1 = nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        self.double_encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.double_encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.double_encoder4 = PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        self.double_encoder5 = PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        self.double_encoder6 = PVMLayer(input_dim=c_list[4], output_dim=c_list[5])

        if bridge:
            self.double_scab = ABM(c_list, split_att)

        self.double_decoder1 = PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        self.double_decoder2 = PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        self.double_decoder3 = PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        self.double_decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)
        self.double_decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)
        self.double_ebn1 = nn.GroupNorm(4, c_list[0])
        self.double_ebn2 = nn.GroupNorm(4, c_list[1])
        self.double_ebn3 = nn.GroupNorm(4, c_list[2])
        self.double_ebn4 = nn.GroupNorm(4, c_list[3])
        self.double_ebn5 = nn.GroupNorm(4, c_list[4])
        self.double_dbn1 = nn.GroupNorm(4, c_list[4])
        self.double_dbn2 = nn.GroupNorm(4, c_list[3])
        self.double_dbn3 = nn.GroupNorm(4, c_list[2])
        self.double_dbn4 = nn.GroupNorm(4, c_list[1])
        self.double_dbn5 = nn.GroupNorm(4, c_list[0])

        self.double_final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out=self.mem(out)
        t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))  # (8,48,8,8)
        t5 = out

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  ##(8,48,8,8) b, c4, H/32, W/32
        out5 = torch.add(out5, t5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # (8,32,16,16) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # (8,24,32,32) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)

        temp = out0.clone().repeat(1, 3, 1, 1)

        double_out = F.gelu(F.max_pool2d(self.double_ebn1(self.double_encoder1(temp)), 2, 2))
        double_t1 = double_out

        double_out = F.gelu(F.max_pool2d(self.double_ebn2(self.double_encoder2(double_out)), 2, 2))
        double_t2 = double_out

        double_out = F.gelu(F.max_pool2d(self.double_ebn3(self.double_encoder3(double_out)), 2, 2))
        double_t3 = double_out

        double_out = F.gelu(F.max_pool2d(self.double_ebn4(self.double_encoder4(double_out)), 2, 2))
        double_t4 = double_out

        double_out = F.gelu(F.max_pool2d(self.double_ebn5(self.double_encoder5(double_out)), 2, 2))
        double_t5 = double_out

        if self.bridge: double_t1, double_t2, double_t3, double_t4, double_t5 = self.double_scab(double_t1, double_t2, double_t3, double_t4, double_t5)

        double_out = F.gelu(self.double_encoder6(double_out))

        double_out5 = F.gelu(self.double_dbn1(self.double_decoder1(double_out)))
        double_out5 = torch.add(double_out5, double_t5)

        double_out4 = F.gelu(F.interpolate(self.double_dbn2(self.double_decoder2(double_out5)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))
        double_out4 = torch.add(double_out4, double_t4)

        double_out3 = F.gelu(F.interpolate(self.double_dbn3(self.double_decoder3(double_out4)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))

        double_out3 = torch.add(double_out3, double_t3)

        double_out2 = F.gelu(F.interpolate(self.double_dbn4(self.double_decoder4(double_out3)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))
        double_out2 = torch.add(double_out2, double_t2)

        double_out1 = F.gelu(F.interpolate(self.double_dbn5(self.double_decoder5(double_out2)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))
        double_out1 = torch.add(double_out1, double_t1)

        double_out0 = F.interpolate(self.double_final(double_out1), scale_factor=(2, 2), mode='bilinear',
                                align_corners=True)

        return torch.sigmoid(out0), torch.sigmoid(double_out0)
