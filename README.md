# SO-VITS-SVC详细安装、训练、推理使用步骤

本帮助文档为项目 [so-vits-svc](https://github.com/MaxMax2016/so-vits-svc) 的详细中文安装、调试、推理教程，您也可以直接选择官方[README](https://github.com/MaxMax2016/so-vits-svc#readme)文档
撰写：Sucial [点击跳转B站主页](https://space.bilibili.com/445022409)

----

## 1. 环境依赖

> - **本项目需要的环境：**
> NVIDIA-CUDA
> Python <= 3.10
> Pytorch
> FFmpeg

### - Cuda

- 在cmd控制台里输入```nvidia-smi.exe```以查看显卡驱动版本和对应的cuda版本

- 前往 [NVIDIA-Developer](https://developer.nvidia.com/) 官网下载与系统**对应**的Cuda版本
以```Cuda-11.7```版本为例（**注：本文下述所有配置均在```Cuda-11.7```下演示**）[Cuda11.7下载地址](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) 根据自己的系统和需求选择安装（一般本地Windows用户请依次选择```Windows```, ```x86_64```, ```系统版本```, ```exe(local)```）
- 安装成功之后在cmd控制台中输入```nvcc -V```, 出现类似以下内容则安装成功：

```shell
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
    Cuda compilation tools, release 11.7, V11.7.64
    Build cuda_11.7.r11.7/compiler.31294372_0
```

#### **特别注意！**

- 目前为止pytorch最高支持到```cuda11.7```
- 如果您在上述第一步中查看到自己的Cuda版本>11.7，请依然选择11.7进行下载安装（Cuda有版本兼容性）并且安装完成后再次在cmd输入```nvidia-smi.exe```并不会出现cuda版本变化，即任然显示的是>11,7的版本
- **Cuda的卸载方法：**打开控制面板-程序-卸载程序，将带有```NVIDIA CUDA```的程序全部卸载即可（一共5个）

### - Python

- 前往 [Python官网](https://www.python.org/) 下载Python，版本需要低于3.10（详细安装方法以及添加Path此处省略，网上随便一查都有）
- 安装完成后在cmd控制台中输入```python```出现类似以下内容则安装成功：

```shell
    Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>> 
```

- 配置python下载镜像源（有国外网络条件可跳过）
在cmd控制台依次执行

```shell
    # 设置清华大学下载镜像
    pip config set global.index-url http://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

#### 安装依赖库

- 在任意位置新建名为```requirements.txt```的文本文件，输入以下内容保存

```shell
    Flask==2.1.2
    Flask_Cors==3.0.10
    gradio==3.4.1
    numpy==1.23.5
    playsound==1.3.0
    PyAudio==0.2.12
    pydub==0.25.1
    pyworld==0.3.2
    requests==2.28.1
    scipy==1.10.0
    sounddevice==0.4.5
    SoundFile==0.10.3.post1
    starlette==0.19.1
    tqdm==4.63.0
    scikit-maad
    praat-parselmouth
    tensorboard
    librosa
```

- 在该文本文件所处文件夹内右击空白处选择 **在终端中打开** 并执行下面命令以安装库（若出现报错请尝试用```pip install [库名称]```重新单独安装直至成功）

```shell
    pip install -r requirements.txt
```

- 接下来我们需要**单独安装**```torch```, ```torchaudio```, ```torchvision```这三个库，下面提供两种方法

#### 方法1（便捷但不建议，因为我在测试这种方法过程中发现有问题，对后续配置AI有影响

> 直接前往 [Pytorch官网](https://pytorch.org/get-started/locally/) 选择所需版本然后复制Run this Command栏显示的命令至cmd安装（不建议）

#### 方法2（较慢但稳定，建议）

- 前往该地址使用```Ctrl+F```搜索直接下载whl包 [点击前往](https://download.pytorch.org/whl/)
>
> - 这个项目需要的是
> ```torch==1.10.0+cu113```
> ```torchaudio==0.10.0+cu113```
> 1.10.0 和0.10.0表示是pytorch版本，cu113表示cuda版本11.3
> 以此类推，请选择**适合自己的版本**安装

- **下面我将以```Cuda11.7```版本为例**
***--示例开始--***
>
> - 我们需要安装以下三个库
>
> 1. ```torch-1.13.0+cu117``` 点击下载：[torch-1.13.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-win_amd64.whl)
其中cp310指```python3.10```, ```win-amd64```表示windows 64位操作系统
> 2. ```torchaudio-0.13.0+cu117```点击下载：[torchaudio-0.13.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torchaudio-0.13.0%2Bcu117-cp310-cp310-win_amd64.whl)
> 3. ```torchvision-0.14.0+cu117```点击下载：[torchvision-0.14.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torchvision-0.14.0%2Bcu117-cp310-cp310-win_amd64.whl)

- 下载完成后进入进入下载的whl文件的目录，在所处文件夹内右击空白处选择 **在终端中打开** 并执行下面命令以安装库

```shell
    pip install .\torch-1.13.0+cu117-cp310-cp310-win_amd64.whl
    # 回车运行(安装时间较长)
    pip install .\torchaudio-0.13.0+cu117-cp310-cp310-win_amd64.whl
    # 回车运行
    pip install .\torchvision-0.14.0+cu117-cp310-cp310-win_amd64.whl 
    # 回车运行
```

- 务必在出现```Successfully installed ...```之后再执行下一条命令，第一个torch包安装时间较长
***--示例结束--***

安装完```torch```, ```torchaudio```, ```torchvision```这三个库之后，在cmd控制台运用以下命令检测cuda与torch版本是否匹配

```shell
    python
    # 回车运行
    import torch
    # 回车运行
    print(torch.__version__)
    # 回车运行
    print(torch.cuda.is_available())
    # 回车运行
```

- 最后一行出现```True```则成功，出现```False```则失败，需要重新安装

### - FFmpeg

- 前往 [FFmpeg官网](https://ffmpeg.org/) 下载。解压至任意位置并在高级系统设置-环境变量中添加Path定位至```.\ffmpeg\bin```（详细安装方法以及添加Path此处省略，网上随便一查都有）
- 安装完成后在cmd控制台中输入```ffmpeg -version```出现类似以下内容则安装成功

```shell
ffmpeg version git-2020-08-12-bb59bdb Copyright (c) 2000-2020 the FFmpeg developers
built with gcc 10.2.1 (GCC) 20200805
configuration: [此处省略一大堆内容]
libavutil      56. 58.100 / 56. 58.100
libavcodec     58.100.100 / 58.100.100
...
```

## 2. 预训练AI

### - 下载项目源码

- 前往 [so-vits-svc](https://github.com/MaxMax2016/so-vits-svc) 选择```32k```分支（本教程针对```32k```）下载源代码。安装了git的可直接git以下地址

- 解压到任意文件夹

### - 下载预训练模型

- 这部分官方文档写得很详细，我这边直接引用

> **hubert**
> <https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt>
> **G与D预训练模型**
> <https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth>
> <https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth>
> **上述三个底模如果原链接下载不了请点击下方的链接**
> <https://pan.baidu.com/s/1uw6W3gOBvMbVey1qt_AzhA?pwd=80eo> 提取码：80eo

- ```hubert-soft-0d54a1f4.pt```放入```.\hubert```文件夹
- ```D_0.pth和G_0.pth```文件放入```.\logs\32k```文件夹

### - 准备训练样本
>
> 准备的训练数据，建议60-100条语音(**格式务必为wav，不同的说话人建立不同的文件夹**)，每条语音控制在**4-8秒！**（确保语音不要有噪音或尽量降低噪音，一个文件夹内语音必须是一个人说的），可以训练出效果不错的模型

- 将语音连带文件夹（有多个人就多个文件夹）一起放入```.\dataset_raw```文件夹里，文件结构类似如下：

```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

- 此外还需要在```.\dataset_raw```文件夹内新建并编辑```config.json```，代码如下：

```shell
"n_speakers": 10    //修改数字为说话人的人数
"spk":{
    "speaker0": 0,  //修改speaker0为第一个说话人的名字，需要和文件夹名字一样，后面的: 0, 不需要改
    "speaker1": 1,  //以此类推
    "speaker2": 2,
    //以此类推
}
```

### - 样本预处理

#### 下面的所有步骤若出现报错请多次尝试，若一直报错就是第一部分环境依赖没有装到位，可以根据报错内容重新安装对应的库。（一般如果正确安装了的话出现报错请多次尝试或者关机重启，肯定可以解决报错的。）

#### 1. 重采样

- 在```so-vits-svc```文件夹内运行终端，直接执行：

```shell
    python resample.py
```

**注意：如果遇到如下报错：**

```shell
...
E:\vs\so-vits-svc-32k\resample.py:17: FutureWarning: Pass sr=None as keyword args. From version 0.10 passing these as positional arguments will result in an error
  wav, sr = librosa.load(wav_path, None)
E:\vs\so-vits-svc-32k\resample.py:17: FutureWarning: Pass sr=None as keyword args. From version 0.10 passing these as positional arguments will result in an error
  wav, sr = librosa.load(wav_path, None)
...
```

请打开```resample.py```，修改第```17```行内容

```shell
# 第17行修改前如下
wav, sr = librosa.load(wav_path, None)
# 第17行修改后如下
wav, sr = librosa.load(wav_path, sr = None)
```

保存，重新执行```python resample.py```命令

- 成功运行后，在```.\dataset\32k```文件夹中会有说话人的wav语音，之后```dataset_raw```文件夹就可以删除了

#### 2. 自动划分训练集，验证集，测试集，自动生成配置文件

- 在```so-vits-svc```文件夹内运行终端，直接执行：

```shell
    python preprocess_flist_config.py
```

- 出现类似以下内容则处理成功：

```shell
PS E:\vs\so-vits-svc-32k> python preprocess_flist_config.py
100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1993.49it/s]
Writing ./filelists/train.txt
100%|██████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:00<?, ?it/s]
Writing ./filelists/val.txt
100%|████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<?, ?it/s]
Writing ./filelists/test.txt
100%|████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<?, ?it/s]
Writing configs/config.json
```

#### 3. 生成hubert和f0

- 在```so-vits-svc```文件夹内运行终端，直接执行：

```shell
    python preprocess_hubert_f0.py
```

- 出现类似以下内容则处理成功：（我这里演示时只用了20条音频）

```shell
PS E:\vs\so-vits-svc-32k> python preprocess_hubert_f0.py
Loading hubert for content...
Loaded hubert.
  0%|                                                                                  | 0/20 [00:00<?, ?it/s]dataset/32k\speaker\1_01.wav
  5%|████                                                                              | 1/20 [00:03<01:00,  3.20s/it]dataset/32k\speaker\1_02.wav
 10%|████████                                                                          | 2/20 [00:03<00:25,  1.40s/it]dataset/32k\speaker\1_03.wav
 15%|████████████                                                                      | 3/20 [00:03<00:14,  1.19it/s]dataset/32k\speaker\1_04.wav
 20%|████████████████▌                                                                 | 4/20 [00:03<00:09,  1.69it/s]dataset/32k\speaker\1_05.wav
 25%|████████████████████                                                              | 5/20 [00:03<00:06,  2.39it/s]dataset/32k\speaker\1_06.wav
 30%|████████████████████████                                                          | 6/20 [00:04<00:04,  2.98it/s]dataset/32k\speaker\1_07.wav
 35%|█████████████████████████████                                                     | 7/20 [00:04<00:03,  3.48it/s]dataset/32k\speaker\1_08.wav
 40%|█████████████████████████████████                                                 | 8/20 [00:04<00:03,  3.78it/s]dataset/32k\speaker\1_09.wav
 45%|█████████████████████████████████████                                             | 9/20 [00:04<00:02,  4.13it/s]dataset/32k\speaker\1_10.wav
 50%|█████████████████████████████████████████                                         | 10/20 [00:04<00:02,  4.41it/s]dataset/32k\speaker\1_11.wav
 55%|█████████████████████████████████████████████                                     | 11/20 [00:04<00:01,  4.71it/s]dataset/32k\speaker\1_12.wav
 60%|█████████████████████████████████████████████████                                 | 12/20 [00:05<00:01,  4.93it/s]dataset/32k\speaker\1_13.wav
 65%|█████████████████████████████████████████████████████                             | 13/20 [00:05<00:01,  5.25it/s]dataset/32k\speaker\1_14.wav
 70%|█████████████████████████████████████████████████████████                         | 14/20 [00:05<00:01,  5.46it/s]dataset/32k\speaker\1_15.wav
 75%|█████████████████████████████████████████████████████████████▌                    | 15/20 [00:05<00:00,  6.19it/s]dataset/32k\speaker\1_16.wav
 80%|█████████████████████████████████████████████████████████████████▌                | 16/20 [00:05<00:00,  5.84it/s]dataset/32k\speaker\1_17.wav
 85%|█████████████████████████████████████████████████████████████████████             | 17/20 [00:06<00:00,  5.43it/s]dataset/32k\speaker\1_18.wav
 90%|█████████████████████████████████████████████████████████████████████████         | 18/20 [00:06<00:00,  5.27it/s]dataset/32k\speaker\1_19.wav
 95%|█████████████████████████████████████████████████████████████████████████████     | 19/20 [00:06<00:00,  5.26it/s]dataset/32k\speaker\1_20.wav
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [00:06<00:00,  3.03it/s]
```

#### 4. 修改配置文件和部分源代码

- 打开上面第二步过程中生成的配置文件```.\configs\config.json```修改第```13```行代码```"batch_size"```的数值。这边解释一下```"batch_size": 12,```数值12要根据自己电脑的显存（任务管理器-GPU-**专用**GPU内存）来调整
>
> - **修改建议**
> 6G显存 建议修改成2或3
> 8G显存 建议修改成4
>"batch_size"参数调小可以解决显存不够的问题
>
- 修改```train.py```

```Python shell
# 第60行将nccl改成gloo（如果后续开始训练时gloo报错就改回nccl，一般不会报错）
# 修改前如下
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
# 修改后如下
    dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)

# 第44行开始
# 修改前如下
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port
#修改后增加代码后如下
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 这里的0代表GPU0是用来训练的显卡，不知道是0还是1的可以在任务管理器查看，如果是双显卡的话一定要选择适合的显卡
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # 这里的32如果懂的话也可以修改，不懂别改
```

## 3. 开始训练

- 在```so-vits-svc```文件夹内运行终端，直接执行下面命令开始训练
**注意：开始训练前建议重启一下电脑清理内存和显存，并且关闭后台游戏，动态壁纸等等软件，最好只留一个cmd窗口**

```shell
    python train.py -c configs/config.json -m 32k
```

- 出现以下报错就是显存不够了

```shell
torch.cuda.OutOfMemoryError: CUDA out of menory. Tried to allocate 16.80 MiB (GPU 0; 8.0 GiB total capacity; 7.11 Gi8 already allocated; 0 bytes free; 7.30 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation. See documentation for Memory Management and PYTORCH_cUDA_ALLOC_CONF
# 注意：一定是 0 bytes free < Tried to allocate 16.80 MiB 才是显存不足，不然就是别的问题
```

- **这边报错可能会比较多，如果出现报错先尝试重新执行```python train.py -c configs/config.json -m 32k```，多重试几遍，或者关机重启，一般是会成功的。如果报错一直是同一个报错，那就是对应的那里出问题了（要靠报错找问题所在）**
- 成功执行以后应该是类似如下内容：

```shell
2023-02-08 18:07:42,439 32k INFO {'train': {'log_interval': 200, 'eval_interval': 1000, 'seed': 1234, 'epochs': 10000, 'learning_rate': 0.0001, 'betas': [0.8, 0.99], 'eps': 1e-09, 'batch_size': 2, 'fp16_run': False, 'lr_decay': 0.999875, 'segment_size': 17920, 'init_lr_ratio': 1, 'warmup_epochs': 0, 'c_mel': 45, 'c_kl': 1.0, 'use_sr': True, 'max_speclen': 384, 'port': '8001'}, 'data': {'training_files': 'filelists/train.txt', 'validation_files': 'filelists/val.txt', 'max_wav_value': 32768.0, 'sampling_rate': 32000, 'filter_length': 1280, 'hop_length': 320, 'win_length': 1280, 'n_mel_channels': 80, 'mel_fmin': 0.0, 'mel_fmax': None}, 'model': {'inter_channels': 192, 'hidden_channels': 192, 'filter_channels': 768, 'n_heads': 2, 'n_layers': 6, 'kernel_size': 3, 'p_dropout': 0.1, 'resblock': '1', 'resblock_kernel_sizes': [3, 7, 11], 'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 'upsample_rates': [10, 8, 2, 2], 'upsample_initial_channel': 512, 'upsample_kernel_sizes': [16, 16, 4, 4], 'n_layers_q': 3, 'use_spectral_norm': False, 'gin_channels': 256, 'ssl_dim': 256, 'n_speakers': 2}, 'spk': {'Sucial': 0}, 'model_dir': './logs\\32k'}
2023-02-08 18:07:42,440 32k WARNING E:\vs\so-vits-svc-32k is not a git repository, therefore hash value comparison will be ignored.
2023-02-08 18:07:45,451 32k INFO Loaded checkpoint './logs\32k\G_0.pth' (iteration 1)
2023-02-08 18:07:45,998 32k INFO Loaded checkpoint './logs\32k\D_0.pth' (iteration 1)
2023-02-08 18:07:55,722 32k INFO Train Epoch: 1 [0%]
2023-02-08 18:07:55,723 32k INFO [1.376741886138916, 3.908522129058838, 12.127800941467285, 35.539894104003906, 4.270486354827881, 0, 0.0001]
2023-02-08 18:08:01,381 32k INFO Saving model and optimizer state at iteration 1 to ./logs\32k\G_0.pth
2023-02-08 18:08:02,344 32k INFO Saving model and optimizer state at iteration 1 to ./logs\32k\D_0.pth
2023-02-08 18:08:19,482 32k INFO ====> Epoch: 1
2023-02-08 18:08:40,093 32k INFO ====> Epoch: 2
2023-02-08 18:09:01,010 32k INFO ====> Epoch: 3
2023-02-08 18:09:21,715 32k INFO ====> Epoch: 4
2023-02-08 18:09:42,242 32k INFO ====> Epoch: 5
2023-02-08 18:10:02,528 32k INFO ====> Epoch: 6
2023-02-08 18:10:22,965 32k INFO ====> Epoch: 7
2023-02-08 18:10:29,149 32k INFO Train Epoch: 8 [14%]
2023-02-08 18:10:29,150 32k INFO [2.378505229949951, 2.3670239448547363, 10.534687042236328, 19.235595703125, 1.8958038091659546, 200, 9.991253280566489e-05]
2023-02-08 18:10:43,388 32k INFO ====> Epoch: 8
2023-02-08 18:11:03,722 32k INFO ====> Epoch: 9
2023-02-08 18:11:23,859 32k INFO ====> Epoch: 10
...
```

- 出现类似以上的内容就说明是在开始训练了（显存会直接爆满）。停止训练有下面两种方法：
>
> 1. 按```Ctrl+C```
> 2. 直接右上角叉掉
> 在控制台中运行 ```python train.py -c config/config.json -m 32k```即可继续训练
>
### - 日志及训练次数的查看

- 日志保存的位置：```.\logs\32k\train.log```
**阅读举例：**

```shell
# 示例3
2023-02-08 18:32:24,942 32k INFO [2.252035617828369, 2.5846095085144043, 8.220404624938965, 5   17.75478744506836, 0.9781494140625, 2000, 9.911637167309565e-05]
2023-02-08 18:32:28,889 32k INFO Saving model and optimizer state at iteration 72 to ./logs\32k\G_2000.pth
2023-02-08 18:32:29,661 32k INFO Saving model and optimizer state at iteration 72 to ./logs\32k\D_2000.pth
# 示例1
2023-02-08 18:32:39,907 32k INFO ====> Epoch: 72
2023-02-08 18:33:00,099 32k INFO ====> Epoch: 73
2023-02-08 18:33:20,682 32k INFO ====> Epoch: 74 
2023-02-08 18:33:40,887 32k INFO ====> Epoch: 75
2023-02-08 18:34:01,460 32k INFO ====> Epoch: 76
2023-02-08 18:34:21,798 32k INFO ====> Epoch: 77
2023-02-08 18:34:41,866 32k INFO ====> Epoch: 78
2023-02-08 18:34:54,712 32k INFO Train Epoch: 79 [57%]
# 示例2
2023-02-08 18:34:54,712 32k INFO [2.282658100128174, 2.5492446422576904, 10.027194023132324, 15.401838302612305, 1.598284363746643, 2200, 9.902967736366644e-05]
```

> **以下的解释我引用了B站up主inifnite_loop的解释，[相关视频](https://www.bilibili.com/video/BV1Bd4y1W7BN) [相关专栏](https://www.bilibili.com/read/cv21425662)**
>
> - 需要关注两个参数：Epoch和global_step
> Epoch表示迭代批次，每一批次可以看作一个迭代分组
> Global_step表示总体迭代次数
> - 两者的关系是global_step = 最多语音说话人的语音数 /  batch_size  * epoch
> batch_size是配置文件中的参数
> - **示例1:** 每一次迭代输出内 ```====> Epoch: 74``` 表示第74迭代批次完成
> - **示例2:** ```Global_step``` 每200次输出一次 （配置文件中的参数```log_interval```）
> - **示例3:** ```Global_step``` 每1000次输出一次（配置文件中的参数```eval_interval```），会保存模型到新的文件
>
#### 一般情况下训练10000次（大约2小时）就能得到一个不错的声音模型了

### - 保存的训练模型
>
> 以上，我们谈论到了每1000次迭代才会保存一次模型样本，那么，这些样本保存在哪里呢？如何处理这些样本呢？下面我将详细讲述。

- 训练模型保存位置：```.\logs\32k```
- 训练一定时间后打开这个路径，你会发现有很多文件：

```shell
D_0.pth
D_1000.pth
D_2000.pth
D_3000.pth
D_4000.pth
...
G_0.pth
G_1000.pth
G_2000.pth
G_3000.pth
G_4000.pth
...
```

- 如果你的硬盘空间不足，那么只要留下最后一次的G和D就可以了，前面的都可以删除（但是不要删别的文件）

## 4. 推理使用
>
> 按上述方法训练得到最后一次的G和D后，该如何使用这些模型呢？下面我将讲述具体的使用操作方法
>
### - 准备干声

- 准备一首歌的干声，干声可以靠软件提取，我这边推荐的是Ultimate Vocal Remover，该软件开源并且可以在Github上下载到。[下载地址](https://github.com/Anjok07/ultimatevocalremovergui)
- 用音频处理软件（如Au，Studio One等）将这个干声分成若干段**不超过40秒**的片段并且一一保存
- 将你处理好的干声片段放入```.\raw```文件夹

### - 修改推理代码

- 打开```inference_main.py```，修改第```17-27```行，具体修改内容如下：

```shell
model_path = "logs/32k/G_10000.pth" # 这里改成你最新训练出来的G模型路径
config_path = "configs/config.json"
svc_model = Svc(model_path, config_path)
infer_tool.mkdir(["raw", "results"])

# 支持多个wav文件，放在raw文件夹下
clean_names = ["vocals_01", "vocals_02","vocals_03"] # 这里修改成你要处理的干声片段的文件名，支持多个文件
trans = [0]  # 音高调整，支持正负（半音）
spk_list = ['Sucial']  # 这里是说话人的名字，之前准备训练样本的文件夹名字
slice_db = -40  # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
wav_format = 'wav'  # 音频输出格式
```
- 如果下一步推理生成时出现错误，请尝试以下修改：(感谢kahotv提供建议) [详细信息](https://github.com/SUC-DriverOld/so-vits-svc-Chinese-Detaild-Documents/issues/1)

```shell
#inference_main.py line35 第35行，
wav_path = Path(raw_audio_path).with_suffix('.wav')
#改为
wav_path = str(Path(raw_audio_path).with_suffix('.wav'))
```

### - 推理生成

- 修改完成后保存代码，在```so-vits-svc```文件夹内运行终端，执行下面命令开始推理生成

```shell
    python .\inference_main.py
```

- 待黑窗口自动关闭后，推理生成完成。生成的音频文件在```.\results```文件夹下
- 如果听上去效果不好，就多训练模型，10000次不够就训练20000次

### - 后期处理

- 将生成的干音和歌曲伴奏（也可以通过Ultimate Vocal Remover提取）导入音频处理软件&宿主软件（如Au，Studio One等）进行混音和母带处理，最终得到成品。

## 5. 感谢名单
>
> - **以下是对本文档的撰写有帮助的感谢名单：**
> so-vits-svc [官方源代码和帮助文档](https://github.com/MaxMax2016/so-vits-svc)
> B站up主inifnite_loop [相关视频](https://www.bilibili.com/video/BV1Bd4y1W7BN) [相关专栏](https://www.bilibili.com/read/cv21425662)
> 所有提供训练音频样本的人员
