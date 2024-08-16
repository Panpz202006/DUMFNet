# DUMFNet

## 0. Abstract

With the increasing adoption of the Mamba framework, state-space models have shown promising results in computer vision. However, these models have not consistently outperformed their CNN-based or Transformer-based counterparts. This limitation is primarily due to their pre-defined scanning schedules, which lack positional awareness and disproportionately emphasize posterior tokens, making the extraction of local region features challenging. Furthermore, most of existing Mamba-based works often overlook the benefits of integrating multiple visual encoding strategies. To address these challenges, we propose a novel DoubleU-Net framework with multiple visual encoding strategies and a local-based scanning mechanism. The comparative 
\& ablation experiments demonstrate that DUMFNet is superior to or competitive with the existing SOTA methods.



## 1. Overview

<img src="https://github.com/Panpz202006/DUMFNet/blob/main/Figs/DUMFNet.png" />



## 2. Main Environments

The environment installation procedure can be followed by [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet), or by following the steps below (python=3.8):

```
conda create -n DUMFNet python=3.8
conda activate DUMFNet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```



## 3. Datasets

You can refer to [UltraLight-VM-UNet's](https://github.com/wurenkai/UltraLight-VM-UNet) approach to processing datasets, or download the .npy files of these datasets from this [link](https://drive.google.com/drive/folders/1aNuwMmOJq8X8gCKOjy6gDar1G5PieoXi), and then organize the .npy files into the following format:

'./datasets/'

- ISIC2017
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy
- ISIC2018
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy
- PH2
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy



## 4. Train the DUMFNet

```
python train.py
```



## 5. Test the DUMFNet 

First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.

```
python test.py
```



## 6. Acknowledgement

Thanks to [Vim](https://github.com/hustvl/Vim), [VM-UNet](https://github.com/JCruan519/VM-UNet) and [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) for their outstanding works.
