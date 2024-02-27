本帮助文档为项目 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 的详细中文安装、调试、推理教程，您也可以直接选择官方[README](https://github.com/svc-develop-team/so-vits-svc#readme)文档
撰写：Sucial [点击跳转 B 站主页](https://space.bilibili.com/445022409)

**写在开头：如需 so-vits-svc3.0 版本的教程，请切换至 [3.0 分支](https://github.com/SUC-DriverOld/so-vits-svc-Chinese-Detaild-Documents/tree/3.0)**

**相关教程和参考资料**
[官方 README 文档](https://github.com/svc-develop-team/so-vits-svc) | [一些报错的解决办法（来自 B 站 up：**羽毛布団**）](https://www.bilibili.com/read/cv22206231) | [本文档配套视频教程](https://www.bilibili.com/video/BV1Hr4y197Cy/) | [UVR5人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/)

**文档的持续完善**：若遇到本文档内未提到的报错，您可以在 issues 中提问；若遇到项目 bug，请给原项目提 issues；想要更加完善这份教程，欢迎来给教程提 pr！

---

# ✅SoftVC VITS Singing Voice Conversion 教程目录

## 最后更新时间：2024.2.26。本次更新结束，文档和教程视频进入【暂停维护】状态。

## 本次更新大改教程文档，建议仔细阅读！

## 点击前往：[本文档配套视频教程](https://www.bilibili.com/video/BV1Hr4y197Cy/) | [UVR5人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/) 注意：配套视频可能较老，仅供参考，一切以最新教程文档为准。

- [✅0. 用前须知](#0-用前须知)
  - [0.0 任何国家，地区，组织和个人使用此项目必须遵守以下法律](#00-任何国家地区组织和个人使用此项目必须遵守以下法律)
  - [0.1 硬件需求](#01-硬件需求)
  - [0.2 提前准备](#02-提前准备)
  - [0.3 训练周期](#03-训练周期)
- [✅1. 环境依赖](#1-环境依赖)
  - [1.1 Cuda](#11-cuda)
  - [1.2 Python](#12-python)
  - [1.3 Pytorch](#13-pytorch)
  - [1.4 安装依赖](#14-安装依赖)
  - [1.5 FFmpeg](#15-ffmpeg)
- [✅2. 配置及训练（参考官方文档）](#2-配置及训练参考官方文档)
  - [2.0 关于兼容 4.0 模型的问题](#20-关于兼容-40-模型的问题)
  - [2.1 关于 Python 版本问题](#21-关于-python-版本问题)
  - [2.2 预先下载的模型文件](#22-预先下载的模型文件)
    - [必须项](#必须项)
    - [可选项(强烈建议使用)](#可选项强烈建议使用)
    - [可选项(根据情况选择)](#可选项根据情况选择)
  - [2.3 数据集准备](#23-数据集准备)
  - [2.4 数据预处理](#24-数据预处理)
    - [2.4.0 音频切片](#240-音频切片)
    - [2.4.1 重采样至 44100Hz 单声道](#241-重采样至-44100hz-单声道)
    - [2.4.2 自动划分训练集、验证集，以及自动生成配置文件](#242-自动划分训练集-验证集以及自动生成配置文件)
    - [2.4.3 生成 hubert 与 f0](#243-生成-hubert-与-f0)
  - [2.5 训练](#25-训练)
    - [2.5.1 扩散模型（可选）](#251-扩散模型可选)
    - [2.5.2 主模型训练](#252-主模型训练)
- [✅3. 推理（参考官方文档）](#3-推理参考官方文档)
  - [3.1 命令行推理](#31-命令行推理)
  - [3.2 WebUI 推理](#32-webui-推理)
- [✅4. 增强效果的可选项](#4-增强效果的可选项)
  - [自动 f0 预测](#自动-f0-预测)
  - [聚类音色泄漏控制](#聚类音色泄漏控制)
  - [特征检索](#特征检索)
- [✅5.其他可选项](#5其他可选项)
  - [5.1 模型压缩](#51-模型压缩)
  - [5.2 声线混合](#52-声线混合)
    - [5.2.1 静态声线混合](#521-静态声线混合)
    - [5.2.2 动态声线混合](#522-动态声线混合)
  - [5.3 Onnx 导出](#53-onnx-导出)
- [✅6. 简单混音处理及成品导出](#6-简单混音处理及成品导出)
- [✅ 附录：常见报错的解决办法](#-附录常见报错的解决办法)
- [✅感谢名单](#感谢名单)

# SoftVC VITS Singing Voice Conversion 教程

# ✅0. 用前须知

## 0.0 任何国家，地区，组织和个人使用此项目必须遵守以下法律

### 《民法典》

#### 第一千零一十九条

任何组织或者个人**不得**以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。**未经**肖像权人同意，**不得**制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。**未经**肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。
**对自然人声音的保护，参照适用肖像权保护的有关规定**

#### 第一千零二十四条

【名誉权】民事主体享有名誉权。任何组织或者个人**不得**以侮辱、诽谤等方式侵害他人的名誉权。

#### 第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》

#### 《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=中华人民共和国刑法)》

#### 《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 本教程仅供交流与学习使用，请勿用于违法违规或违反公序良德等不良用途

#### 出于对音源提供者的尊重请勿用于鬼畜用途

### 0.0.1. 继续使用视为已同意本教程所述相关条例，本教程已进行劝导义务，不对后续可能存在问题负责。

1. 本教程内容**仅代表个人**，均不代表 so-vits-svc 团队及原作者观点
2. 本教程涉及到的开源代码请自行**遵守其开源协议**
3. 本教程默认使用由**so-vits-svc 团队维护**的仓库
4. 若制作视频发布，**推荐注明**使用项目的**Github**链接，tag**推荐**使用**so-vits-svc**以便和其他基于技术进行区分
5. 云端训练和推理部分可能涉及资金使用，如果你是**未成年人**，请在**获得监护人的许可与理解后**进行，未经许可引起的后续问题，本教程**概不负责**
6. 本地训练（尤其是在硬件较差的情况下）可能需要设备长时间**高负荷**运行，请做好设备养护和散热措施
7. 请确保你制作数据集的数据来源**合法合规**，且数据提供者明确你在制作什么以及可能造成的后果
8. 出于设备原因，本教程仅在**Windows**系统下进行过测试，Mac 和 Linux 请确保自己有一定解决问题能力
9. 该项目为**歌声合成**项目，**无法**进行其他用途，请知悉

### 0.0.2. 声明

本项目为开源、离线的项目，SvcDevelopTeam 的所有成员与本项目的所有开发者以及维护者（以下简称贡献者）对本项目没有控制力。本项目的贡献者从未向任何组织或个人提供包括但不限于数据集提取、数据集加工、算力支持、训练支持、推理等一切形式的帮助；本项目的贡献者不知晓也无法知晓使用者使用该项目的用途。故一切基于本项目训练的 AI 模型和合成的音频都与本项目贡献者无关。一切由此造成的问题由使用者自行承担。

此项目完全离线运行，不能收集任何用户信息或获取用户输入数据。因此，这个项目的贡献者不知道所有的用户输入和模型，因此不负责任何用户输入。

本项目只是一个框架项目，本身并没有语音合成的功能，所有的功能都需要用户自己训练模型。同时，这个项目没有任何模型，任何二次分发的项目都与这个项目的贡献者无关。

### 0.0.3. 使用规约

### Warning：请自行解决数据集授权问题，禁止使用非授权数据集进行训练！任何由于使用非授权数据集进行训练造成的问题，需自行承担全部责任和后果！与仓库、仓库维护者、svc develop team、教程发布者 无关

1. 本项目是基于学术交流目的建立，仅供交流与学习使用，并非为生产环境准备。
2. 任何发布到视频平台的基于 sovits 制作的视频，都必须要在简介明确指明用于变声器转换的输入源歌声、音频，例如：使用他人发布的视频 / 音频，通过分离的人声作为输入源进行转换的，必须要给出明确的原视频、音乐链接；若使用是自己的人声，或是使用其他歌声合成引擎合成的声音作为输入源进行转换的，也必须在简介加以说明。
3. 由输入源造成的侵权问题需自行承担全部责任和一切后果。使用其他商用歌声合成软件作为输入源时，请确保遵守该软件的使用条例，注意，许多歌声合成引擎使用条例中明确指明不可用于输入源进行转换！
4. 禁止使用该项目从事违法行为与宗教、政治等活动，该项目维护者坚决抵制上述行为，不同意此条则禁止使用该项目。
5. 继续使用视为已同意本仓库 README 所述相关条例，本仓库 README 已进行劝导义务，不对后续可能存在问题负责。
6. 如果将此项目用于任何其他企划，请提前联系并告知本仓库作者，十分感谢。

## 0.1 硬件需求

1. 推理目前分为**命令行推理**和**WebUI 推理**，对速度要求不高的话 CPU 和 GPU 均可使用
2. 至少需要**6G 以上**显存的**NVIDIA 显卡**（如 RTX3060）
3. 云端一般常见的为 V100（16G）、V100（32G）、A100（40G）、A100（80G）等显卡，部分云端提供 RTX3090 等显卡

## 0.2 提前准备

1. **至少**准备 200 条 8s（约 30 分钟**持续说话**时长，即约 1.5 小时**正常说话**采样）左右时长的**干净**人声（**无底噪，无混响**）作为训练集。并且最好保持说话者**情绪起伏波动较小**，人声**响度合适**，并且做好**响度匹配**
2. 请提前准备训练需要用到的**底模**（**挺重要的**）
3. **须知**：歌声作为训练集**只能**用来推理歌声，但语音作为训练集即可以推理歌声，也可以用来生成 TTS。但用语音作为训练集可能使**高音和低音推理出现问题**（即缺少高低音训练样本），有一种可行的解决方法是模型融合。
4. 推理：需准备**底噪<30dB**，尽量**不要带过多混响和和声**的**干音**进行推理
5. **须知**：推理女声歌曲时，建议用女声训练模型，同理男声也类似

## 0.3 训练周期

在**有底模**的前提下，选取**500 条音频**作为训练集，经多次测试（RTX3060 Laptop,专用显存6G， `batch_size = 3`）得到以下结论：

1. 模型训练步数 10w+（若每晚训练约 8 小时，需要约 3 晚+）
2. 模型训练步数 2w-3w（若每晚训练约 8 小时，需要约 1 晚）
3. 模型训练步数 5w-8w（若每晚训练约 8 小时，需要约 2-3 晚）

**模型怎样才算训练好了**？

1. 这是一个非常无聊且没有意义的问题。就好比上来就问老师我家孩子怎么才能学习好，除了你自己，没有人能回答这个问题。
2. 模型的训练关联于你的数据集质量、时长，所选的编码器、f0 算法，甚至一些超自然的玄学因素，即便你有一个成品模型，最终的转换效果也要取决于你的输入源以及推理参数。这不是一个线性的的过程，之间的变量实在是太多，所以你非得问“为什么我的模型出来不像啊”、“模型怎样才算训练好了”这样的问题，我只能说 WHO F**KING KNOWS?
3. 但也不是一点办法没有，只能烧香拜佛了。我不否认烧香拜佛当然是一个有效的手段，但你也可以借助一些科学的工具，例如 Tensorboard 等，下一段就将教你怎么通过看 Tensorboard 来辅助了解训练状态，当然，最强的辅助工具其实长在你自己身上，一个声学模型怎样才算训练好了? 塞上耳机，让你的耳朵告诉你吧

# ✅1. 环境依赖

**本项目需要的环境**：NVIDIA-CUDA | Python = 3.8.9 | Pytorch | FFmpeg
**注意：现已添加conda环境配置文件Sovits.yaml，会使用conda的可以通过该配置文件一件配置环境**（环境使用torch：2.0.1+cu117），**请从code处下载**。

## 1.1 Cuda

- 在 cmd 控制台里输入`nvidia-smi.exe`以查看显卡驱动版本和对应的 cuda 版本

- 前往 [NVIDIA-Developer](https://developer.nvidia.com/) 官网下载与系统**对应**的 Cuda 版本
- **此处强烈建议CUDA版本选择11.7，测试下来最稳定**

  以`Cuda-11.7`版本为例（**注：本文下述所有配置均在`Cuda-11.7`下演示**）[Cuda11.7 下载地址](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) 根据自己的系统和需求选择安装（一般本地 Windows 用户请依次选择`Windows`, `x86_64`, `系统版本`, `exe(local)`）


- 安装成功之后在 cmd 控制台中输入`nvcc -V`, 出现类似以下内容则安装成功：

```shell
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
    Cuda compilation tools, release 11.7, V11.7.64
    Build cuda_11.7.r11.7/compiler.31294372_0
```

#### **特别注意！**

- Cuda需要与下方 1.3 Pytorch版本相匹配
- 卸载方法：打开控制面板-程序-卸载程序，将带有`NVIDIA CUDA`的程序全部卸载即可（一共 5 个）

## 1.2 Python

- 前往 [Python 官网](https://www.python.org/) 下载 Python3.8.9（若使用conda配置python遇到没有3.8.9版本也可以直接输入3.8）详细安装方法以及添加 Path 此处省略，网上随便一查都有）
- 安装完成后在 cmd 控制台中输入`python`出现类似以下内容则安装成功：

```shell
    Python 3.8.9（tags/v3.8.9:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>>
```

**注：关于 Python 版本问题**

在进行测试后，我们认为Python 3.8.9能够稳定地运行该项目
(但不排除高版本也可以运行)

- 配置 python 下载镜像源（有国外网络条件可跳过）
  在 cmd 控制台依次执行

```shell
    # 设置清华大学下载镜像
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

- 如果想要还原为默认源，类似的，你仅需要在控制台执行

```shell
pip config set global.index-url https://pypi.python.org/simple
```

- 以下是一些国内常用的镜像源

**python国内镜像源**
- 清华: https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里云: https://mirrors.aliyun.com/pypi/simple/
- 中国科技大学: https://pypi.mirrors.ustc.edu.cn/simple/
- 华中科技大学: http://pypi.hustunique.com/
- 山东理工大学: http://pypi.sdutlinux.org/

```shell
# 临时更换
pip install package -i https://pypi.tuna.tsinghua.edu.cn/simple
# 永久更换
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1.3 Pytorch

- 经过多次实验测得，pytorch2.0.1+cu117为稳定版本，所以此处**强烈建议安装CUDA11.7的Pytorch**

- 首先我们需要**单独安装**`torch`, `torchaudio`, `torchvision`这三个库，直接前往 [Pytorch 官网](https://pytorch.org/get-started/locally/) 选择所需版本然后复制 Run this Command 栏显示的命令至 cmd 安装
- 如需手动指定`torch`的版本在其后面添加版本号即可，例如`…… torch==2.1.0 ……`
- 由于版本更新，11.7的Pytorch可能复制不到下载链接，此时可以复制下方的安装命令进行安装。

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

- 安装完`torch`, `torchaudio`, `torchvision`这三个库之后，在 cmd 控制台运用以下命令检测 cuda 与 torch 版本是否匹配

```shell
    python
    # 回车运行
    import torch
    # 回车运行
    print(torch.cuda.is_available())
    # 回车运行
```

- 最后一行出现`True`则成功，出现`False`则失败，需要重新安装

## 1.4 安装依赖

- 在项目文件夹内右击空白处选择 **在终端中打开** 并执行下面命令以安装库（若出现报错请尝试用`pip install [库名称]`重新单独安装直至成功）
- **注意，项目文件夹内含有三个 requirements 的 txt 分别对应不同系统和需求，请根据需求选择其中一个**（没什么特殊需求并且是 windows 系统的话选 requirements_win.txt）

```shell
    pip install -r requirements_win.txt
```

确保安装正确无误后请更新以下三个依赖：

```shell
pip install --upgrade fastapi==0.84.0
pip install --upgrade gradio==3.41.2
pip install --upgrade pydantic==1.10.12
```

### 关于 fairseq 安装不了的问题，windows 的解决方案如下

- 第一步：更新 pip 到最新版
- 第二步：安装 visual studio 2022，社区版就行，然后组件里装“使用 c++的桌面开发”。全部安装完成之后再重新 pip install farseq 即可完成安装

## 1.5 FFmpeg

- 前往 [FFmpeg 官网](https://ffmpeg.org/) 下载。解压至任意位置并在高级系统设置-环境变量中添加 Path 定位至`.\ffmpeg\bin`（详细安装方法以及添加 Path 此处省略，网上随便一查都有）
- 安装完成后在 cmd 控制台中输入`ffmpeg -version`出现类似以下内容则安装成功

```shell
ffmpeg version git-2020-08-12-bb59bdb Copyright (c) 2000-2020 the FFmpeg developers
built with gcc 10.2.1 (GCC) 20200805
configuration: [此处省略一大堆内容]
libavutil      56. 58.100 / 56. 58.100
libavcodec     58.100.100 / 58.100.100
...
```

# ✅2. 配置及训练（参考官方文档）

## 2.0 关于兼容 4.0 模型的问题

- 可通过修改 4.0 模型的 config.json 对 4.0 的模型进行支持，需要在 config.json 的 model 字段中添加 speech_encoder 字段，具体见下

```
  "model": {
    .........
    "ssl_dim": 256,
    "n_speakers": 200,
    "speech_encoder":"vec256l9"
  }
```

## 2.1 关于 Python 版本问题

在进行测试后，我们认为`Python 3.8.9`能够稳定地运行该项目
(但不排除高版本也可以运行)

配置及训练

## 2.2 预先下载的模型文件

#### **必须项**

**以下编码器需要选择一个使用**

##### **1. 若使用 contentvec 作为声音编码器（推荐）**

`vec768l12`与`vec256l9` 需要该编码器

- contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  - 放在`pretrain`目录下

或者下载下面的 ContentVec，大小只有 199MB，但效果相同:

- contentvec ：[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
  - 将文件名改为`checkpoint_best_legacy_500.pt`后，放在`pretrain`目录下

```shell
# contentvec
wget -P pretrain/ http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt
# 也可手动下载放在pretrain目录
```

##### **2. 若使用 hubertsoft 作为声音编码器**

- soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  - 放在`pretrain`目录下

##### **3. 若使用 Whisper-ppg 作为声音编码器**

- 下载模型 [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 该模型适配`whisper-ppg`
- 下载模型 [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), 该模型适配`whisper-ppg-large`
  - 放在`pretrain`目录下

##### **4. 若使用 cnhubertlarge 作为声音编码器**

- 下载模型 [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)
  - 放在`pretrain`目录下

##### **5. 若使用 dphubert 作为声音编码器**

- 下载模型 [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)
  - 放在`pretrain`目录下

##### **6. 若使用 OnnxHubert/ContentVec 作为声音编码器**

- 下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
  - 放在`pretrain`目录下

#### **编码器列表**

- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"

#### **可选项(强烈建议使用)**

- 预训练底模文件： `G_0.pth` `D_0.pth`

  - 放在`logs/44k`目录下

- 扩散模型预训练底模文件： `model_0.pt`
  - 放在`logs/44k/diffusion`目录下

扩散模型引用了[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)的 Diffusion Model，底模与[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)的扩散模型底模通用，可以去[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)获取扩散模型的底模

虽然底模一般不会引起什么版权问题，但还是请注意一下，比如事先询问作者，又或者作者在模型描述中明确写明了可行的用途

**提供 4.1 训练底模，需自行下载（需具备外网条件）**:
- 下载地址1：官方Huggingface下载 [D_0.pth](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_D_320000.pth) [G_0.pth](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_G_320000.pth) [model_0.pt](https://huggingface.co/datasets/ms903/Diff-SVC-refactor-pre-trained-model/resolve/main/fix_pitch_add_vctk_600k/model_0.pt)
- 下载地址2：[点击跳转下载](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model) 包含扩散模型训练底模
- 下载地址3：[百度网盘转存](https://pan.baidu.com/s/17IlNuphFHAntLklkMNtagg?pwd=dkp9) 此处不更新，随时可能落后

**提供 3.0 训练底模，需自行下载**
- 下载地址：[百度网盘转存](https://pan.baidu.com/s/1uw6W3gOBvMbVey1qt_AzhA?pwd=80eo)

#### **可选项(根据情况选择)**

##### NSF-HIFIGAN

如果使用`NSF-HIFIGAN增强器`或`浅层扩散`的话，需要下载预训练的 NSF-HIFIGAN 模型，如果不需要可以不下载

- 预训练的 NSF-HIFIGAN 声码器 ：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
  - 解压后，将四个文件放在`pretrain/nsf_hifigan`目录下
 
##### RMVPE

如果使用`rmvpe`F0预测器的话，需要下载预训练的 RMVPE 模型

- 下载模型[rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，目前首推该权重。
  - 解压缩`rmvpe.zip`，并将其中的`model.pt`文件改名为`rmvpe.pt`并放在`pretrain`目录下

##### FCPE(预览版)

- [FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/MelPE)是由svc-develop-team自主研发的一款全新的F0预测器，是一个为实时语音转换所设计的专用F0预测器，他将在未来成为Sovits实时语音转换的首选F0预测器.（论文未来会有的）

如果使用 `fcpe` F0预测器的话，需要下载预训练的 FCPE 模型

- 下载模型 [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
  - 放在`pretrain`目录下


## 2.3 数据集准备

仅需要以以下文件结构将数据集放入 dataset_raw 目录即可

```
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

可以自定义说话人名称

```
dataset_raw
└───suijiSUI
    ├───1.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

## 2.4 数据预处理

### 2.4.0 音频切片

将音频切片至`5s - 15s`, 稍微长点也无伤大雅，实在太长可能会导致训练中途甚至预处理就爆显存

可以使用[audio-slicer-GUI](https://github.com/flutydeer/audio-slicer)、[audio-slicer-CLI](https://github.com/openvpi/audio-slicer)

一般情况下只需调整其中的`Minimum Interval`，普通陈述素材通常保持默认即可，歌唱素材可以调整至`100`甚至`50`

切完之后手动删除过长过短的音频

**如果你使用 Whisper-ppg 声音编码器进行训练，所有的切片长度必须小于 30s**

### 2.4.1 重采样至 44100Hz 单声道

```shell
python resample.py
```

#### 注意

虽然本项目拥有重采样、转换单声道与响度匹配的脚本 resample.py，但是默认的响度匹配是匹配到 0db。这可能会造成音质的受损。而 python 的响度匹配包 pyloudnorm 无法对电平进行压限，这会导致爆音。所以建议可以考虑使用专业声音处理软件如`adobe audition`等软件做响度匹配处理。若已经使用其他软件做响度匹配，可以在运行上述命令时添加`--skip_loudnorm`跳过响度匹配步骤。如：

```shell
python resample.py --skip_loudnorm
```

### 2.4.2 自动划分训练集、验证集，以及自动生成配置文件

```shell
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder 拥有七个选择

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
whisper-ppg-large
cnhubertlarge
dphubert
```

如果省略 speech_encoder 参数，默认值为 vec768l12

**使用响度嵌入**

若使用响度嵌入，需要增加`--vol_aug`参数，比如：

```shell
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

使用后训练出的模型将匹配到输入源响度，否则为训练集响度。

#### 此时可以在生成的 config.json 与 diffusion.yaml 修改部分参数

##### config.json

* `keep_ckpts`：训练时保留最后几个模型，`0`为保留所有，默认只保留最后`3`个

* `all_in_mem`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用

* `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可

* `vocoder_name` : 选择一种声码器，默认为`nsf-hifigan`.

##### diffusion.yaml

* `cache_all_data`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用

* `duration`：训练时音频切片时长，可根据显存大小调整，**注意，该值必须小于训练集内音频的最短时间！**

* `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可

* `timesteps` : 扩散模型总步数，默认为 1000.

* `k_step_max` : 训练时可仅训练`k_step_max`步扩散以节约训练时间，注意，该值必须小于`timesteps`，0 为训练整个扩散模型，**注意，如果不训练整个扩散模型将无法使用仅扩散模型推理！**

##### **声码器列表**

```
nsf-hifigan
nsf-snake-hifigan
```

### 2.4.3 生成 hubert 与 f0

```shell
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor 拥有四个选择

```
crepe
dio
pm
harvest
rmvpe
fcpe
```

如果训练集过于嘈杂，请使用 crepe 处理 f0

如果省略 f0_predictor 参数，默认值为 rmvpe

尚若需要浅扩散功能（可选），需要增加--use_diff 参数，比如

```shell
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除 dataset_raw 文件夹了

## 2.5 训练

### 2.5.1 扩散模型（可选）

尚若需要浅扩散功能，需要训练扩散模型，扩散模型训练方法为:

```shell
python train_diff.py -c configs/diffusion.yaml
```

### 2.5.2 主模型训练

```shell
python train.py -c configs/config.json -m 44k
```

模型训练结束后，模型文件保存在`logs/44k`目录下，扩散模型在`logs/44k/diffusion`下

# ✅3. 推理（参考官方文档）

## 3.1 命令行推理

使用 inference_main.py

```shell
# 例
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

必填项部分：

- `-m` | `--model_path`：模型路径
- `-c` | `--config_path`：配置文件路径
- `-n` | `--clean_names`：wav 文件名列表，放在 raw 文件夹下
- `-t` | `--trans`：音高调整，支持正负（半音）
- `-s` | `--spk_list`：合成目标说话人名称
- `-cl` | `--clip`：音频强制切片，默认 0 为自动切片，单位为秒/s

可选项部分：部分具体见下一节

- `-lg` | `--linear_gradient`：两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值 0，单位为秒
- `-f0p` | `--f0_predictor`：选择 F0 预测器,可选择 crepe,pm,dio,harvest,默认为 pm(注意：crepe 为原 F0 使用均值滤波器)
- `-a` | `--auto_predict_f0`：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
- `-cm` | `--cluster_model_path`：聚类模型或特征检索索引路径，如果没有训练聚类或特征检索则随便填
- `-cr` | `--cluster_infer_ratio`：聚类方案或特征检索占比，范围 0-1，若没有训练聚类模型或特征检索则默认 0 即可
- `-eh` | `--enhance`：是否使用 NSF_HIFIGAN 增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
- `-shd` | `--shallow_diffusion`：是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN 增强器将会被禁止
- `-usm` | `--use_spk_mix`：是否使用角色融合/动态声线融合
- `-lea` | `--loudness_envelope_adjustment`：输入源响度包络替换输出响度包络融合比例，越靠近 1 越使用输出响度包络
- `-fr` | `--feature_retrieval`：是否使用特征检索，如果使用聚类模型将被禁用，且 cm 与 cr 参数将会变成特征检索的索引路径与混合比例

浅扩散设置：

- `-dm` | `--diffusion_model_path`：扩散模型路径
- `-dc` | `--diffusion_config_path`：扩散模型配置文件路径
- `-ks` | `--k_step`：扩散步数，越大越接近扩散模型的结果，默认 100
- `-od` | `--only_diffusion`：纯扩散模式，该模式不会加载 sovits 模型，以扩散模型推理
- `-se` | `--second_encoding`：二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差

### 注意

如果使用`whisper-ppg` 声音编码器进行推理，需要将`--clip`设置为 25，`-lg`设置为 1。否则将无法正常推理。

## 3.2 WebUI 推理

使用以下命令打开 webui 界面，推理参数参考 3.1

```shell
python webUI.py
```

# ✅4. 增强效果的可选项

如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用(这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显)

### 自动 f0 预测

4.0 模型训练过程会训练一个 f0 预测器，对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动的，但转换歌声时请不要启用此功能！！！会严重跑调！！

- 在 inference_main 中设置 auto_predict_f0 为 true 即可

### 聚类音色泄漏控制

介绍：聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点

使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低

- 训练过程：
  - 使用 cpu 性能较好的机器训练
  - 执行`python cluster/train_cluster.py`，模型的输出会在`logs/44k/kmeans_10000.pt`
  - 聚类模型目前可以使用 gpu 进行训练，执行`python cluster/train_cluster.py --gpu`
 
```shell
# CPU
python cluster/train_cluster.py
# GPU
python cluster/train_cluster.py --gpu
```

- 推理过程：
  - `inference_main.py`中指定`cluster_model_path`
  - `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可

### 特征检索

介绍：跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比，

- 训练过程：
  首先需要在生成 hubert 与 f0 后执行：

```shell
python train_index.py -c configs/config.json
```

模型的输出会在`logs/44k/feature_and_index.pkl`

- 推理过程：
  - 需要首先制定`--feature_retrieval`，此时聚类方案会自动切换到特征检索方案
  - `inference_main.py`中指定`cluster_model_path` 为模型输出文件
  - `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可

# ✅5.其他可选项

## 5.1 模型压缩

生成的模型含有继续训练所需的信息。如果确认不再训练，可以移除模型中此部分信息，得到约 1/3 大小的最终模型。

使用 compress_model.py

```shell
# 例
python compress_model.py -c="configs/config.json" -i="logs/44k/G_30400.pth" -o="logs/44k/release.pth"
```

## 5.2 声线混合

### 5.2.1 静态声线混合

**参考`webUI.py`文件中，小工具/实验室特性的静态声线融合。**

介绍:该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线
**注意：**

1. 该功能仅支持单说话人的模型
2. 如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个 SpaekerID 下的声音
3. 保证所有待混合模型的 config.json 中的 model 字段是相同的
4. 输出的混合模型可以使用待合成模型的任意一个 config.json，但聚类模型将不能使用
5. 批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
6. 混合比例调整建议大小在 0-100 之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
7. 混合完毕后，文件将会保存在项目根目录中，文件名为 output.pth
8. 凸组合模式会将混合比例执行 Softmax 使混合比例相加为 1，而线性组合模式不会

### 5.2.2 动态声线混合

**参考`spkmix.py`文件中关于动态声线混合的介绍**

角色混合轨道 编写规则：

角色 ID : \[\[起始时间 1, 终止时间 1, 起始数值 1, 起始数值 1], [起始时间 2, 终止时间 2, 起始数值 2, 起始数值 2]]

起始时间和前一个的终止时间必须相同，第一个起始时间必须为 0，最后一个终止时间必须为 1 （时间的范围为 0-1）

全部角色必须填写，不使用的角色填\[\[0., 1., 0., 0.]]即可

融合数值可以随便填，在指定的时间段内从起始数值线性变化为终止数值，内部会自动确保线性组合为 1（凸组合条件），可以放心使用

推理的时候使用`--use_spk_mix`参数即可启用动态声线混合

## 5.3 Onnx 导出

使用 onnx_export.py

- 新建文件夹：`checkpoints` 并打开
- 在`checkpoints`文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如`aziplayer`
- 将你的模型更名为`model.pth`，配置文件更名为`config.json`，并放置到刚才创建的`aziplayer`文件夹下
- 将 onnx_export.py 中`path = "NyaruTaffy"` 的 `"NyaruTaffy"` 修改为你的项目名称，`path = "aziplayer" (onnx_export_speaker_mix，为支持角色混合的onnx导出)`
- 运行 onnx_export.py
- 等待执行完毕，在你的项目文件夹下会生成一个`model.onnx`，即为导出的模型

注意：Hubert Onnx 模型请使用 MoeSS 提供的模型，目前无法自行导出（fairseq 中 Hubert 有不少 onnx 不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出 shape 和结果都有问题）

# ✅6. 简单混音处理及成品导出

### 使用音频宿主软件处理推理后音频，具体流程比较麻烦，请参考 [配套视频教程](https://www.bilibili.com/video/BV1Hr4y197Cy/) | [UVR5人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/) 或其他更专业的混音教程。

# ✅ 附录：常见报错的解决办法

**部分报错及解决方法，来自羽毛布団大佬https://www.bilibili.com/read/cv22206231**

## 关于爆显存

如果你在终端或 WebUI 界面的报错中出现了这样的报错:

```shell
OutOfMemoryError: CUDA out of memory.Tried to allocate XX GiB (GPU O: XX GiB total capacity; XX GiB already allocated; XX MiB Free: XX GiB reserved in total by PyTorch)
```

不要怀疑，你的显卡显存或虚拟内存不够用了。以下是100%解决问题的解决方法，照着做必能解决。请不要再在各种地方提问这个问题了

1. 在报错中找到 XX GiB already allocated 之后，是否显示 0 bytes free，如果是 0 bytes free 那么看第2，3，4步，如果显示 XX MiB free 或者 XX GiB free，看第 5 步
2. 如果是预处理的时候爆显存:
  a. 换用对显存占用友好的 f0 预测器 (友好度从高到低: pm >= harvest >= rmvpe ≈ fcpe >> crepe)，建议首选 rmvpe 或fcpe
  b. 多进程预处理改为1
3. 如果是训练的时候爆显存
  a. 检查数据集有没有过长的切片 (20秒以上）
  b. 调小批量大小 (batch size)
  c. 更换一个占用低的项目
  d. 去 AutoDL 等云算力平台上面租一张大显存的显卡跑
4. 如果是推理的时候爆显存:
  a. 推理源 (千声) 不干净 (有残留的混响，伴奏，和声)，导致自动切片切不开。提取干声最佳实践请参考[UVR5歌曲人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/)
  b. 调大切片闽值 (比如-40调成-30，再大就不建议了，你也不想唱一半就被切一刀吧)
  c. 设置强制切片，从60秒开始尝试，每次减小10秒，直到能成功推理
  d. 使用 cpu 推理，速度会很慢但是不会爆显存
5. 如果显示仍然有空余显存却还是爆显存了，是你的虚拟内存不够大，调整到至少 50G 以上

## 安装依赖时出现的相关报错

**1. 依赖找不到导致的无法安装**

出现**类似**以下报错时：

```shell
ERROR: Could not find a version that satisfies the requirement librosa==0.9.1 (from versions: none)
ERROR: No matching distribution found for librosa==0.9.1
# 主要特征是
No matching distribution found for xxxxx
Could not find a version that satisfies the requirement xxxx
```

具体解决方法为：更换安装源。手动安装这一依赖时添加下载源，以下是两个常用的镜像源地址

- 清华大学：https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里云：http://mirrors.aliyun.com/pypi/simple

使用`pip install [包名称] -i [下载源地址]`，例如我想在阿里源下载 librosa 这个依赖，并且要求依赖版本是 0.9.1，那么应该在 cmd 中输入以下命令：

```shell
pip install librosa==0.9.1 -i http://mirrors.aliyun.com/pypi/simple
```

**2. 报错ERROR: Package 'networkx' requires a different Python: 3.8.9 not in '>=3.9**

此报错的原因是因为torch官方更新，使得当前的torch版本太新导致的，解决方法两个：

1. 升级python至3.9（但可能造成不稳定）
2. 降低torch版本（建议）也可理解为降低CUDA版本，比如我目前使用的是2.0.1+cu117。

注意：**如果你在之前已经配置好了环境并且能用了，请忽略此条提醒**

## 数据集预处理和模型训练时的相关报错

**1. 报错：`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0 in position xx`**
答：数据集文件名中不要包含中文或日文等非西文字符，特别注意**中文**括号，逗号，冒号，分号，引号等等都是不行的。改完名字**一定要**重新预处理，然后再进行训练！！！

**2. 报错：`The expand size of the tensor (768) must match the existing size (256) at non-singleton dimension 0.`**
答：把 dataset/44k 下的内容全部删了，重新走一遍预处理流程


## 主模型训练时出现的相关报错

**1. 报错：RuntimeError: DataLoader worker (pid(s) 13920) exited unexpectedly**

```shell
raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
RuntimeError: DataLoader worker (pid(s) 13920) exited unexpectedly
```

解决方法：调小 batchsize 值，调大虚拟内存，重启电脑清理显存，直到 batchsize 值和虚拟内存合适不报错为止

**2. 报错：`torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with exit code 3221225477`**
解决方法：调大虚拟内存，管理员运行cmd

**3. 报错：`AssertionError: CPU training is not allowed.`**
没有解决方法：非 N 卡跑不了。（也不是完全跑不了，但如果你是纯萌新的话，那我的回答确实就是：跑不了）

**4. 报错：页面文件太小，无法完成操作。**
解决方法：调大虚拟内存大小，具体的方法各种地方一搜就能搜到，不展开了。

## 使用WebUI时相关报错

**1. 出现以下报错时**：

- 启动 webUI 时报错：`ImportError: cannot import name 'Schema' from 'pydantic'`
- webUI 加载模型时报错：`AttributeError("'Dropdown' object has no attribute 'update'")`
- **凡是报错中涉及到 fastapi, gradio, pydantic 这三个依赖的报错**

**解决方法如下**：
需限制部分依赖版本，在安装完`requirements_win.txt`后，在 cmd 中依次输入以下命令以更新依赖包：

```shell
pip install --upgrade fastapi==0.84.0
pip install --upgrade gradio==3.41.2
pip install --upgrade pydantic==1.10.12
```

**2. 报错：`Given groups=1, weight of size [xxx, 256, xxx], expected input[xxx, 768, xxx] to have 256 channels, but got 768 channels instead`**
或**报错: 配置文件中的编码器与模型维度不匹配**
解决方法：v1 分支的模型用了 vec768 的配置文件，如果上面报错的 256 的 768 位置反过来了那就是 vec768 的模型用了 v1 的配置文件。检查配置文件中的”ssl_dim”一项，如果这项是256，那你的speech encoder应当修改为“vec256|9”，如果是768，则是"vec768|12"

**3. 报错：`'HParams' object has no attribute 'xxx'`**
解决方法：无法找到音色，一般是配置文件和模型没对应，打开配置文件拉到最下面看看有没有你训练的音色

----

# 感谢名单：

- so-vits-svc [官方源代码和帮助文档](https://github.com/svc-develop-team/so-vits-svc)
- B 站 up 主 inifnite_loop [相关视频](https://www.bilibili.com/video/BV1Bd4y1W7BN) [相关专栏](https://www.bilibili.com/read/cv21425662)
- 一些报错的解决办法[（B 站 up 主：**羽毛布団** 相关专栏）](https://www.bilibili.com/read/cv22206231)
- 所有提供训练音频样本的人员
