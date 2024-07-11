import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
import math
from mamba_ssm import Mamba


class LSK_DWConv(nn.Module):
    def __init__(self, dim=768):
        super(LSK_DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class LSK_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = LSK_DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSK_LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class LSK_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSK_LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LSK_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = LSK_Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LSK_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # x = x.clone().permute(0, 3, 1, 2).contiguous()
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        # return x.clone().permute(0, 2, 3, 1).contiguous()
        return x


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


class SC_Att_Bridge(nn.Module):
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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
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


class SCAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1, -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        out = ori_x * attn

        return out


class DUMFNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.Encoder_conv_1 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Encoder_attn_1 = Attention(dim=24, num_heads=8)
        self.Encoder_attn_conv_1 = nn.Conv2d(24, 32, kernel_size=1)
        self.Encoder_mlp_1 = MLP(dim1=24, dim2=32)
        p = 1
        self.p = p * 2
        # self.lsk_blcok_1_1 = LSK_Block(dim=24)
        # self.lsk_blcok_1_2 = LSK_Block(dim=24)
        # self.lsk_blcok_2_1 = LSK_Block(dim=24)
        # self.lsk_blcok_2_2 = LSK_Block(dim=24)

        # self.encoder4_window = nn.Sequential(
        #     PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        # )
        # self.encoder4_random = nn.Sequential(
        #     PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        # )
        # self.sc_attn = SCAttn(in_channels=32)
        # self.sc_attn_2 = SCAttn(in_channels=32)
        # self.sc_attn_3 = SCAttn(in_channels=32)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
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

        self.my_encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.my_encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.my_encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.my_encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.my_encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.my_encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.my_scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.my_decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.my_decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.my_decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.my_decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.my_decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.my_ebn1 = nn.GroupNorm(4, c_list[0])
        self.my_ebn2 = nn.GroupNorm(4, c_list[1])
        self.my_ebn3 = nn.GroupNorm(4, c_list[2])
        self.my_ebn4 = nn.GroupNorm(4, c_list[3])
        self.my_ebn5 = nn.GroupNorm(4, c_list[4])
        self.my_dbn1 = nn.GroupNorm(4, c_list[4])
        self.my_dbn2 = nn.GroupNorm(4, c_list[3])
        self.my_dbn3 = nn.GroupNorm(4, c_list[2])
        self.my_dbn4 = nn.GroupNorm(4, c_list[1])
        self.my_dbn5 = nn.GroupNorm(4, c_list[0])

        self.my_final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

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

    def forward(self, x):  # (8,3,256,256)

        # ori=x.clone()
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8
        # out = self.lsk_blcok_1_1(out)

        b, c, h, w = out.shape

        # image = out.clone()  # 替换为实际的图像数据
        #
        # B, C, H, W = image.size()
        # image_flat = image.view(B, C, -1)
        # num_elements = image_flat.size(2)
        # random_indices = torch.randperm(num_elements)
        # image_shuffled = image_flat[:, :, random_indices]
        # image_shuffled = image_shuffled.view(B, C, H, W)

        window_mamba = out.clone()
        B, C, H, W = window_mamba.shape
        x_div = window_mamba.reshape(B, C, H // self.p, self.p, W // self.p, self.p).permute(0, 3, 5, 1, 2,
                                                                                             4).contiguous().view(
            B * self.p * self.p, C, H // self.p, W // self.p)

        mamba_1 = self.encoder4(x_div).reshape(B, 32, H, W)
        encorder_conv_1 = self.Encoder_conv_1(out)
        attn_out = out.clone().view(b, h * w, -1)
        encorder_attn_1 = self.Encoder_attn_1(attn_out)
        encorder_attn_1 = encorder_attn_1.view(b, 24, h, w)
        encorder_attn_1 = self.Encoder_attn_conv_1(encorder_attn_1)
        mlp_out = out.clone().view(b, h * w, -1)
        encorder_mlp_1 = self.Encoder_mlp_1(mlp_out)
        encorder_mlp_1 = encorder_mlp_1.view(b, 32, h, w)
        out = mamba_1 + encorder_conv_1 + encorder_mlp_1 + encorder_attn_1
        out = F.avg_pool2d(self.ebn4(out), 2, 2)
        softmax_out = torch.softmax(out.clone(), dim=1)
        out = out * softmax_out
        out = F.gelu(out)
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))  # (8,48,8,8)
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  ##(8,48,8,8) b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # (8,32,16,16) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # (8,24,32,32) # b, c2, H/8, W/8
        # out3 = self.lsk_blcok_1_2(out3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        temp = out0.clone().repeat(1, 3, 1, 1)
        # temp = temp*ori+ori
        my_out = F.gelu(F.max_pool2d(self.my_ebn1(self.my_encoder1(temp)), 2, 2))
        my_t1 = my_out  # b, c0, H/2, W/2

        my_out = F.gelu(F.max_pool2d(self.my_ebn2(self.my_encoder2(my_out)), 2, 2))
        my_t2 = my_out  # b, c1, H/4, W/4

        my_out = F.gelu(F.max_pool2d(self.my_ebn3(self.my_encoder3(my_out)), 2, 2))
        my_t3 = my_out  # b, c2, H/8, W/8
        # my_out = self.lsk_blcok_2_1(my_out)

        my_out = F.gelu(F.max_pool2d(self.my_ebn4(self.my_encoder4(my_out)), 2, 2))
        my_t4 = my_out  # b, c3, H/16, W/16

        my_out = F.gelu(F.max_pool2d(self.my_ebn5(self.my_encoder5(my_out)), 2, 2))
        my_t5 = my_out  # b, c4, H/32, W/32

        if self.bridge: my_t1, my_t2, my_t3, my_t4, my_t5 = self.my_scab(my_t1, my_t2, my_t3, my_t4, my_t5)

        my_out = F.gelu(self.my_encoder6(my_out))  # b, c5, H/32, W/32

        my_out5 = F.gelu(self.my_dbn1(self.my_decoder1(my_out)))  # b, c4, H/32, W/32
        my_out5 = torch.add(my_out5, my_t5)  # b, c4, H/32, W/32
        # my_out5 = torch.add(my_out5, t5)  # b, c4, H/32, W/32

        my_out4 = F.gelu(F.interpolate(self.my_dbn2(self.my_decoder2(my_out5)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # b, c3, H/16, W/16
        my_out4 = torch.add(my_out4, my_t4)  # b, c3, H/16, W/16
        # my_out4 = torch.add(my_out4, t4)  # b, c3, H/16, W/16

        my_out3 = F.gelu(F.interpolate(self.my_dbn3(self.my_decoder3(my_out4)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # b, c2, H/8, W/8
        # my_out3 = self.lsk_blcok_2_2(my_out3)
        my_out3 = torch.add(my_out3, my_t3)  # b, c2, H/8, W/8
        # my_out3 = torch.add(my_out3, t3)  # b, c2, H/8, W/8

        my_out2 = F.gelu(F.interpolate(self.my_dbn4(self.my_decoder4(my_out3)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # b, c1, H/4, W/4
        my_out2 = torch.add(my_out2, my_t2)  # b, c1, H/4, W/4
        # my_out2 = torch.add(my_out2, t2)  # b, c1, H/4, W/4

        my_out1 = F.gelu(F.interpolate(self.my_dbn5(self.my_decoder5(my_out2)), scale_factor=(2, 2), mode='bilinear',
                                       align_corners=True))  # b, c0, H/2, W/2
        my_out1 = torch.add(my_out1, my_t1)  # b, c0, H/2, W/2
        # my_out1 = torch.add(my_out1, t1)  # b, c0, H/2, W/2

        my_out0 = F.interpolate(self.my_final(my_out1), scale_factor=(2, 2), mode='bilinear',
                                align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0), torch.sigmoid(my_out0)
