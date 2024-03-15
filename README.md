<div align="center">

# SoftVC VITS Singing Voice Conversion 本地部署教程

**最后更新时间：2024.3.9，本次更新重构教程，文档和教程视频进入【暂停维护】状态**

本帮助文档为项目 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 的详细中文安装、调试、推理教程，您也可以直接选择官方[README](https://github.com/svc-develop-team/so-vits-svc#readme)文档

撰写：Sucial | 点击跳转 [Bilibili 主页](https://space.bilibili.com/445022409)

</div>

---

✨ **我写的一个 so-vits-svc 一键配置环境，启动 webUI 的脚本：[so-vits-svc-webUI-QuickStart-bat](https://github.com/SUC-DriverOld/so-vits-svc-webUI-QuickStart-bat) 欢迎使用！**

✨ **点击前往：[配套视频教程](https://www.bilibili.com/video/BV1Hr4y197Cy/) | [UVR5 人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/)（注意：配套视频可能较老，仅供参考，一切以最新教程文档为准！）**

✨ **相关资料：[官方 README 文档](https://github.com/svc-develop-team/so-vits-svc) | [羽毛布団：一些报错的解决办法](https://www.bilibili.com/read/cv22206231) | [羽毛布団：常见报错解决方法](https://www.yuque.com/umoubuton/ueupp5/ieinf8qmpzswpsvr) | [羽毛布団](https://space.bilibili.com/3493141443250876)大佬的整合包**

> [!NOTE]
>
> **【写在开头】** 不想手动配置环境/伸手党/寻找整合包的，请使用 [羽毛布団](https://space.bilibili.com/3493141443250876) 大佬的整合包
>
> **关于旧版本教程**：如需 so-vits-svc3.0 版本的教程，请切换至 [3.0 分支](https://github.com/SUC-DriverOld/so-vits-svc-Chinese-Detaild-Documents/tree/3.0)。此分支教程已停止更新！
>
> **文档的持续完善**：若遇到本文档内未提到的报错，您可以在 issues 中提问；若遇到项目 bug，请给原项目提 issues；想要更加完善这份教程，欢迎来给提 pr！

# 教程目录

- [0. 用前须知](#0-用前须知)
  - [0.1 使用规约](#01-使用规约)
  - [0.2 硬件需求](#02-硬件需求)
  - [0.3 提前准备](#03-提前准备)
- [1. 环境依赖](#1-环境依赖)
  - [1.1 Cuda](#11-cuda)
  - [1.2 Python](#12-python)
  - [1.3 Pytorch](#13-pytorch)
  - [1.4 其他依赖项安装](#14-其他依赖项安装)
  - [1.5 FFmpeg](#15-ffmpeg)
- [2. 配置及训练](#2-配置及训练)
  - [2.1 关于兼容 4.0 模型的问题](#21-关于兼容-40-模型的问题)
  - [2.2 预先下载的模型文件](#22-预先下载的模型文件)
    - [2.2.1 必须项](#221-必须项)
    - [2.2.2 预训练底模 (强烈建议使用)](#222-预训练底模-强烈建议使用)
    - [2.2.3 可选项 (根据情况选择)](#223-可选项-根据情况选择)
  - [2.3 数据集准备](#23-数据集准备)
  - [2.4 数据预处理](#24-数据预处理)
    - [2.4.0 音频切片](#240-音频切片)
    - [2.4.1 重采样至 44100Hz 单声道](#241-重采样至-44100hz-单声道)
    - [2.4.2 自动划分训练集、验证集，以及自动生成配置文件](#242-自动划分训练集验证集以及自动生成配置文件)
    - [2.4.3 配置文件按需求修改](#243-配置文件按需求修改)
    - [2.4.3 生成 hubert 与 f0](#243-生成-hubert-与-f0)
  - [2.5 训练](#25-训练)
    - [2.5.1 扩散模型（可选）](#251-扩散模型可选)
    - [2.5.2 主模型训练（必须）](#252-主模型训练必须)
    - [2.5.3 Tensorboard](#253-tensorboard)
- [3. 推理](#3-推理)
  - [3.1 命令行推理](#31-命令行推理)
  - [3.2 webUI 推理](#32-webui-推理)
- [4. 增强效果的可选项](#4-增强效果的可选项)
  - [4.1 自动 f0 预测](#41-自动-f0-预测)
  - [4.2 聚类音色泄漏控制](#42-聚类音色泄漏控制)
  - [4.3 特征检索](#43-特征检索)
  - [4.4 声码器微调](#44-声码器微调)
  - [4.5 各模型保存的目录](#45-各模型保存的目录)
- [5.其他可选功能](#5其他可选功能)
  - [5.1 模型压缩](#51-模型压缩)
  - [5.2 声线混合](#52-声线混合)
    - [5.2.1 静态声线混合](#521-静态声线混合)
    - [5.2.2 动态声线混合](#522-动态声线混合)
  - [5.3 Onnx 导出](#53-onnx-导出)
- [6. 简单混音处理及成品导出](#6-简单混音处理及成品导出)
- [附录：常见报错的解决办法](#附录常见报错的解决办法)
  - [关于爆显存](#关于爆显存)
  - [安装依赖时出现的相关报错](#安装依赖时出现的相关报错)
  - [数据集预处理和模型训练时的相关报错](#数据集预处理和模型训练时的相关报错)
  - [使用 WebUI 时相关报错](#使用-webui-时相关报错)
- [感谢名单](#感谢名单)

# 0. 用前须知

### 任何国家，地区，组织和个人使用此项目必须遵守以下法律

#### 《民法典》

#### 第一千零一十九条

任何组织或者个人**不得**以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。**未经**肖像权人同意，**不得**制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。**未经**肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。
**对自然人声音的保护，参照适用肖像权保护的有关规定**

#### 第一千零二十四条

【名誉权】民事主体享有名誉权。任何组织或者个人**不得**以侮辱、诽谤等方式侵害他人的名誉权。

#### 第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》|《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=中华人民共和国刑法)》|《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》|《[中华人民共和国合同法](http://www.npc.gov.cn/zgrdw/npc/lfzt/rlyw/2016-07/01/content_1992739.htm)》

## 0.1 使用规约

> [!WARNING]
>
> 1. **本教程仅供交流与学习使用，请勿用于违法违规或违反公序良德等不良用途。出于对音源提供者的尊重请勿用于鬼畜用途**
> 2. **继续使用视为已同意本教程所述相关条例，本教程已进行劝导义务，不对后续可能存在问题负责**
> 3. **请自行解决数据集授权问题，禁止使用非授权数据集进行训练！任何由于使用非授权数据集进行训练造成的问题，需自行承担全部责任和后果！与仓库、仓库维护者、svc develop team、教程发布者无关！**

具体使用规约如下：

- 本教程内容仅代表个人，均不代表 so-vits-svc 团队及原作者观点
- 本教程默认使用由 so-vits-svc 团队维护的仓库，涉及到的开源代码请自行遵守其开源协议
- 任何发布到视频平台的基于 sovits 制作的视频，都必须要在简介明确指明用于变声器转换的输入源歌声、音频，例如：使用他人发布的视频或音频，通过分离的人声作为输入源进行转换的，必须要给出明确的原视频、音乐链接；若使用是自己的人声，或是使用其他歌声合成引擎合成的声音作为输入源进行转换的，也必须在简介加以说明。
- 请确保你制作数据集的数据来源合法合规，且数据提供者明确你在制作什么以及可能造成的后果。由输入源造成的侵权问题需自行承担全部责任和一切后果。使用其他商用歌声合成软件作为输入源时，请确保遵守该软件的使用条例。注意，许多歌声合成引擎使用条例中明确指明不可用于输入源进行转换！
- 云端训练和推理部分可能涉及资金使用，如果你是未成年人，请在获得监护人的许可与理解后进行，未经许可引起的后续问题，本教程概不负责
- 本地训练（尤其是在硬件较差的情况下）可能需要设备长时间高负荷运行，请做好设备养护和散热措施
- 出于设备原因，本教程仅在 Windows 系统下进行过测试，Mac 和 Linux 请确保自己有一定解决问题能力
- 继续使用视为已同意本仓库 README 所述相关条例，本仓库 README 已进行劝导义务，不对后续可能存在问题负责。

## 0.2 硬件需求

1. 训练**必须使用** GPU 进行训练！推理目前分为**命令行推理**和**WebUI 推理**，对速度要求不高的话 CPU 和 GPU 均可使用。
2. 如需自己训练，请准备至少 **6G 以上专用显存的 NVIDIA 显卡**。
3. 请确保电脑的虚拟内存设置到**30G 以上**，并且最好设置在固态硬盘，不然会很慢。
4. **云端训练建议使用 [AutoDL](https://www.autodl.com/home) 平台**。常见的显卡有： V100（16G）、V100（32G）、A100（40G）、A100（80G）、RTX3090、RTX4080、RTX4090 等显卡。

## 0.3 提前准备

1. **至少**准备约 150 条 8s（约 30 分钟**持续说话**时长，即约 1.0 小时**正常说话**）左右时长的**干净**人声（**无底噪，无混响**）作为训练集。并且最好保持说话者**情绪起伏波动较小**，人声**响度合适**，并且做好**响度匹配**。
2. 请提前下载训练需要用到的**底模**（**挺重要的**）
3. 推理：需准备**底噪<30dB**，尽量**不要带过多混响和和声**的**干音**进行推理

> [!NOTE]
>
> **须知 1**：歌声和语音都可以作为训练集，但语音作为训练集可能使**高音和低音推理出现问题（俗称音域问题/哑音）**（即缺少高低音训练样本），你可以尝试在训练集中**掺杂歌声和语音**，效果会略有提升，但最好的方法是**补充高音训练数据**。
>
> **须知 2**：推理女声歌曲时，建议用女声训练模型，同理男声也类似（否则你可能根本唱不上去或唱不下来，此时只能升调或降调推理）。
>
> **✨ 2024.3.8 最新建议 ✨**：目前 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 项目的 TTS 与 so-vits-svc 的文字转语音相比，训练集需求量更小，训练速度更快，效果更好，所以此处建议若想使用语音生成功能，请移步 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。也因此，建议大家使用歌声素材作为训练集来训练本项目。

# 1. 环境依赖

✨ **本项目需要的环境**：[NVIDIA-CUDA](https://developer.nvidia.com/cuda-toolkit) | [Python](https://www.python.org/) = 3.8.9（项目建议此版本） | [Pytorch](https://pytorch.org/get-started/locally/) | [FFmpeg](https://ffmpeg.org/)

✨ **提醒：现已添加 conda 环境配置文件 Sovits.yaml，会使用 conda 的可以通过该配置文件一件配置环境**（环境使用 torch：2.0.1+cu117），**请从 code 处下载**。

✨ **我写的一个 so-vits-svc 一键配置环境，启动 webUI 的脚本：[so-vits-svc-webUI-QuickStart-bat](https://github.com/SUC-DriverOld/so-vits-svc-webUI-QuickStart-bat) 也可以尝试使用！**

## 1.1 Cuda

- 在 cmd 控制台里输入 `nvidia-smi.exe` 以查看显卡驱动版本和对应的 cuda 版本

- 前往 [NVIDIA-CUDA](https://developer.nvidia.com/cuda-toolkit) 官网下载与系统**对应**的 Cuda 版本
- 以`Cuda-11.7`版本为例（本文下述所有配置均使用`Cuda-11.7`）[点击此处下载 cuda11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) 根据自己的系统和需求选择安装（一般本地 Windows 用户请依次选择`Windows`, `x86_64`, `系统版本`, `exe(local)`）

- 安装成功之后在 cmd 控制台中输入`nvcc -V`, 出现类似以下内容则安装成功：

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

> [!NOTE]
>
> 1. `nvidia-smi.exe` 中显示的 CUDA 版本号具有向下兼容性。例如我显示的是 12.4，但我可以安装 <=12.4 的任意版本 CUDA 版本进行安装。**但此处测试下来，需要选择 >=11.7 的版本进行安装**。
> 2. CUDA 需要与下方 1.3 Pytorch 版本相匹配（其实不适配也问题不大，**只要能被调用到就行**）
> 3. CUDA 卸载方法：打开控制面板-程序-卸载程序，将带有 `NVIDIA CUDA` 的程序全部卸载即可（一共 5 个）

## 1.2 Python

- 前往 [Python 官网](https://www.python.org/) 下载 Python3.8.9 安装并添加系统环境变量。（若使用 conda 配置 python 遇到没有 3.8.9 版本也可以直接选择 3.8）详细安装方法以及添加 Path 此处省略，网上随便一查都有，不再赘述。

```bash
# conda配置方法, 将YOUR_ENV_NAME替换成你想要创建的虚拟环境名字。
conda create -n YOUR_ENV_NAME python=3.8 -y
conda activate YOUR_ENV_NAME
```

- 安装完成后在 cmd 控制台中输入`python`出现类似以下内容则安装成功：

```bash
Python 3.8.9 (tags/v3.8.9:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

**关于 Python 版本问题**：在进行测试后，我们认为 Python 3.8.9 能够稳定地运行该项目(但不排除高版本也可以运行)。

- 配置 python 下载镜像源（有国外网络条件可跳过）
  在 cmd 控制台执行

```bash
# 设置清华大学下载镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

- 如果想要还原为默认源，类似的，你仅需要在控制台执行

```bash
pip config set global.index-url https://pypi.python.org/simple
```

**python 国内常用镜像源**：

- 清华: <https://pypi.tuna.tsinghua.edu.cn/simple>
- 阿里云: <https://mirrors.aliyun.com/pypi/simple/>

```bash
# 临时更换
pip install PACKAGE_NAME -i https://pypi.tuna.tsinghua.edu.cn/simple
# 永久更换
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1.3 Pytorch

- 经过多次实验测得，pytorch2.0.1+cu117 为稳定版本，高版本请自己尝试。

- 首先我们需要 **单独安装** `torch`, `torchaudio`, `torchvision` 这三个库，直接前往 [Pytorch 官网](https://pytorch.org/get-started/locally/) 选择所需版本然后复制 Run this Command 栏显示的命令至控制台安装即可。

- 安装完 `torch`, `torchaudio`, `torchvision` 这三个库之后，在 cmd 控制台运用以下命令检测 torch 能否成功调用 CUDA。最后一行出现 `True` 则成功，出现`False` 则失败，需要重新安装正确的版本。

```bash
python
# 回车运行
import torch
# 回车运行
print(torch.cuda.is_available())
# 回车运行
```

> [!NOTE]
>
> 1. 如需手动指定 `torch` 的版本在其后面添加版本号即可，例如 `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117`
> 2. 安装 CUDA=11.7 的 Pytorch 时，可能会遇到报错 `ERROR: Package 'networkx' requires a different Python: 3.8.9 not in '>=3.9`。此时，请先 `pip install networkx==3.0` 之后再进行 Pytorch 的安装。
> 3. 由于版本更新，11.7 的 Pytorch 可能复制不到下载链接，此时你可以直接复制下方的安装命令进行安装。

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

## 1.4 其他依赖项安装

✨ 在开始其他依赖项安装之前，**请务必下载并安装** [Visual Studio 2022](https://visualstudio.microsoft.com/) 或者 [Microsoft C++ 生成工具](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)（体积较前者更小）。**勾选并安装组件包：“使用 C++的桌面开发”**，执行修改并等待其安装完成。

- 在项目文件夹内右击空白处选择 **在终端中打开** 。使用下面的命令先更新一下 `pip`, `wheel`, `setuptools` 这三个包。

```bash
pip install --upgrade pip wheel setuptools
```

- 执行下面命令以安装库（**若出现报错请多次尝试直到没有报错，依赖全部安装完成**）。注意，项目文件夹内含有三个 requirements 的 txt ，此处选择 `requirements_win.txt`）

```bash
pip install -r requirements_win.txt
```

- 确保安装 **正确无误** 后请使用下方命令更新 `fastapi`, `gradio`, `pydantic` 这三个依赖：

```bash
pip install --upgrade fastapi==0.84.0
pip install --upgrade pydantic==1.10.12
pip install --upgrade gradio==3.41.2
```

## 1.5 FFmpeg

- 前往 [FFmpeg 官网](https://ffmpeg.org/) 下载。解压至任意位置并在高级系统设置-环境变量中添加 Path 。定位至 `.\ffmpeg\bin`（详细安装方法以及添加 Path 此处省略，网上随便一查都有，不再赘述）
- 安装完成后在 cmd 控制台中输入 `ffmpeg -version` 出现类似以下内容则安装成功：

```bash
ffmpeg version git-2020-08-12-bb59bdb Copyright (c) 2000-2020 the FFmpeg developers
built with gcc 10.2.1 (GCC) 20200805
configuration: [此处省略一大堆内容]
libavutil      56. 58.100 / 56. 58.100
libavcodec     58.100.100 / 58.100.100
...[此处省略一大堆内容]
```

# 2. 配置及训练

✨ 此部分内容是整个教程文档中最重要的部分，本教程参考了 [官方文档](https://github.com/svc-develop-team/so-vits-svc#readme)，并适当添加了一些解释和说明，便于理解。

✨ 在开始第二部分内容前，请确保电脑的虚拟内存设置到**30G 以上**，并且最好设置在固态硬盘。具体设置方法请自行上网搜索。

## 2.1 关于兼容 4.0 模型的问题

- 你可以通过修改 4.0 模型的 config.json 对 4.0 的模型进行支持。需要在 config.json 的 model 字段中添加 speech_encoder 字段，具体如下：

```bash
  "model":
  {
    # 省略其他内容

    # "ssl_dim"，填256或者768，需要和下面"speech_encoder"匹配
    "ssl_dim": 256,
    # 说话人个数
    "n_speakers": 200,
    # 或者"vec768l12"，但请注意此项的值要和上面的"ssl_dim"相互匹配。即256对应vec256l9，768对应vec768l12。
    "speech_encoder":"vec256l9"
    # 如果不知道自己的模型是vec768l12还是vec256l9，可以看"gin_channels"字段的值来确认。

    # 省略其他内容
  }
```

## 2.2 预先下载的模型文件

### 2.2.1 必须项

> [!WARNING]
>
> **以下编码器必须需选择一个使用：**
>
> - "vec768l12"
> - "vec256l9"
> - "vec256l9-onnx"
> - "vec256l12-onnx"
> - "vec768l9-onnx"
> - "vec768l12-onnx"
> - "hubertsoft-onnx"
> - "hubertsoft"
> - "whisper-ppg"
> - "cnhubertlarge"
> - "dphubert"
> - "whisper-ppg-large"

**1. 若使用 contentvec 作为声音编码器（推荐）**

`vec768l12`与`vec256l9` 需要该编码器

- 下载 contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)。放在`pretrain`目录下。

或者下载下面的 ContentVec，大小只有 199MB，但效果相同

- contentvec ：[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)。**将文件名改为`checkpoint_best_legacy_500.pt`后**，放在`pretrain`目录下。

**2. 若使用 hubertsoft 作为声音编码器**

- soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)。放在`pretrain`目录下。

**3. 若使用 Whisper-ppg 作为声音编码器**

- 下载模型 [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 该模型适配`whisper-ppg`
- 下载模型 [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), 该模型适配`whisper-ppg-large`
- 放在`pretrain`目录下。

**4. 若使用 cnhubertlarge 作为声音编码器**

- 下载模型 [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)。放在`pretrain`目录下。

**5. 若使用 dphubert 作为声音编码器**

- 下载模型 [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)。放在`pretrain`目录下。

**6. 若使用 OnnxHubert/ContentVec 作为声音编码器**

- 下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)。放在`pretrain`目录下。

#### 各编码器的详解

| 编码器名称            | 优点                                 | 缺点                 |
| --------------------- | ------------------------------------ | -------------------- |
| `vec768l12`（最推荐） | 最还原音色、有大型底模、支持响度嵌入 | 咬字能力较弱         |
| `vec256l9`            | 貌似举不出特别的优点                 | 不支持扩散模型       |
| `hubertsoft`          | 咬字能力较强                         | 音色泄露             |
| `whisper-ppg`         | 咬字最强                             | 音色泄露、显存占用高 |

### 2.2.2 预训练底模 (强烈建议使用)

- 预训练底模文件： `G_0.pth` `D_0.pth`。放在`logs/44k`目录下。

- 扩散模型预训练底模文件： `model_0.pt`。放在`logs/44k/diffusion`目录下。

扩散模型引用了[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)的 Diffusion Model，底模与[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)的扩散模型底模通用，可以去[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)获取扩散模型的底模。

虽然底模一般不会引起什么版权问题，但还是请注意一下，比如事先询问作者，又或者作者在模型描述中明确写明了可行的用途。

> [!NOTE]
>
> **提供 4.1 训练底模，需自行下载（需具备外网条件）**
>
> - 下载地址 1：官方 Huggingface 下载 | [D_0.pth](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_D_320000.pth) | [G_0.pth](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_G_320000.pth) | [model_0.pt](https://huggingface.co/datasets/ms903/Diff-SVC-refactor-pre-trained-model/resolve/main/fix_pitch_add_vctk_600k/model_0.pt) | 下载完成后请对应并重命名为`G_0.pth` `D_0.pth`和`model_0.pt`。
> - 下载地址 2：Huggingface 转存 [【点击跳转下载】](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model) 此处包含扩散模型训练底模。
> - 下载地址 3：[【百度网盘转存】](https://pan.baidu.com/s/17IlNuphFHAntLklkMNtagg?pwd=dkp9) 此处不更新，已经落后。
>
> **提供 3.0 训练底模，需自行下载**
>
> - 下载地址：[【百度网盘转存】](https://pan.baidu.com/s/1uw6W3gOBvMbVey1qt_AzhA?pwd=80eo)

### 2.2.3 可选项 (根据情况选择)

**1. NSF-HIFIGAN**

如果使用`NSF-HIFIGAN增强器`或`浅层扩散`的话，需要下载预训练的 NSF-HIFIGAN 模型，如果不需要可以不下载。

- 预训练的 NSF-HIFIGAN 声码器 ：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
- 解压后，将四个文件放在`pretrain/nsf_hifigan`目录下。

**2. RMVPE**

如果使用`rmvpe`F0 预测器的话，需要下载预训练的 RMVPE 模型。

- 下载模型[rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，目前首推该权重。
- 解压缩`rmvpe.zip`，并将其中的`model.pt`文件改名为`rmvpe.pt`并放在`pretrain`目录下。

**3. FCPE(预览版)**

[FCPE(Fast Context-base Pitch Estimator)](https://github.com/CNChTu/MelPE)是由 svc-develop-team 自主研发的一款全新的 F0 预测器，是一个为实时语音转换所设计的专用 F0 预测器，他将在未来成为 Sovits 实时语音转换的首选 F0 预测器。

如果使用 `fcpe` F0 预测器的话，需要下载预训练的 FCPE 模型。

- 下载模型 [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
- 放在`pretrain`目录下。

## 2.3 数据集准备

1. 按照以下文件结构将数据集放入 dataset_raw 目录。

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

2. 可以自定义说话人名称。

```
dataset_raw
└───suijiSUI
    ├───1.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

## 2.4 数据预处理

### 2.4.0 音频切片

- 将音频切片至`5s - 15s`, 稍微长点也无伤大雅，实在太长可能会导致训练中途甚至预处理就爆显存。

- 可以使用 [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer) | [audio-slicer-CLI](https://github.com/openvpi/audio-slicer) 进行辅助切片。一般情况下只需调整其中的 `Minimum Interval`，普通说话素材通常保持默认即可，歌唱素材可以调整至 `100` 甚至 `50`。

- 切完之后请手动删除过长或过短的音频。

> [!WARNING]
>
> **如果你使用 Whisper-ppg 声音编码器进行训练，所有的切片长度必须小于 30s**

### 2.4.1 重采样至 44100Hz 单声道

使用下面的命令（若已经经过响度匹配，请跳过该行看下面的 NOTE）：

```bash
python resample.py
```

> [!NOTE]
>
> 虽然本项目拥有重采样、转换单声道与响度匹配的脚本 `resample.py`，但是默认的响度匹配是匹配到 0db。这可能会造成音质的受损。而 `python` 的响度匹配包 `pyloudnorm` 无法对电平进行压限，这会导致爆音。所以建议可以考虑使用专业声音处理软件如 `adobe audition` 等软件做响度匹配处理。若已经使用其他软件做响度匹配，可以在运行上述命令时添加 `--skip_loudnorm` 跳过响度匹配步骤。如：

```bash
python resample.py --skip_loudnorm
```

### 2.4.2 自动划分训练集、验证集，以及自动生成配置文件

使用下面的命令（若需要响度嵌入，请跳过该行看下面的使用响度嵌入）：

```bash
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder 拥有以下七个选择，具体讲解请看 **[2.2.1 必须项及各编码器的详解](#各编码器的详解)**。如果省略 speech_encoder 参数，默认值为 vec768l12。

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
whisper-ppg-large
cnhubertlarge
dphubert
```

#### 使用响度嵌入

- 若使用响度嵌入，需要增加`--vol_aug`参数，比如：

```bash
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

- 使用后训练出的模型将匹配到输入源响度，否则为训练集响度。

### 2.4.3 配置文件按需求修改

#### config.json

- `vocoder_name`: 选择一种声码器，默认为`nsf-hifigan`
- `log_interval`：多少步输出一次日志，默认为 `200`
- `eval_interval`：多少步进行一次验证并保存一次模型，默认为 `800`
- `epochs`：训练总轮数，默认为 `10000`，达到此轮数后将自动停止训练
- `learning_rate`：学习率，建议保持默认值不要改
- `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
- `all_in_mem`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用
- `keep_ckpts`：训练时保留最后几个模型，`0`为保留所有，默认只保留最后`3`个

**声码器列表**

```
nsf-hifigan
nsf-snake-hifigan
```

#### diffusion.yaml

- `cache_all_data`：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 **远大于** 数据集体积时可以启用
- `duration`：训练时音频切片时长，可根据显存大小调整，**注意，该值必须小于训练集内音频的最短时间！**
- `batch_size`：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
- `timesteps` : 扩散模型总步数，默认为 1000。完整的高斯扩散一共 1000 步
- `k_step_max` : 训练时可仅训练 `k_step_max` 步扩散以节约训练时间，注意，该值必须小于`timesteps`，0 为训练整个扩散模型，**注意，如果不训练整个扩散模型将无法使用仅扩散模型推理！**

### 2.4.3 生成 hubert 与 f0

使用下面的命令（若需要训练浅扩散，请跳过该行看下面的浅扩散）：

```bash
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor 拥有四个选择，部分 f0 预测器需要额外下载预处理模型，具体请参考 **[2.2.3 可选项 (根据情况选择)](#223-可选项-根据情况选择)**

```
crepe
dio
pm
harvest
rmvpe
fcpe
```

#### 各个 f0 预测器的优缺点

| 预测器  | 优点                                                      | 缺点                                         |
| ------- | --------------------------------------------------------- | -------------------------------------------- |
| pm      | 速度快，占用低                                            | 容易出现哑音                                 |
| crepe   | 基本不会出现哑音                                          | 显存占用高，自带均值滤波，因此可能会出现跑调 |
| dio     | -                                                         | 可能跑调                                     |
| harvest | 低音部分有更好表现                                        | 其他音域就不如别的算法了                     |
| rmvpe   | 六边形战士，目前最完美的预测器                            | 几乎没有缺点（极端长低音可能会出错）         |
| fcpe    | SVC 开发组自研，目前最快的预测器，且有不输 crepe 的准确度 | -                                            |

> [!NOTE]
>
> 1. 如果训练集过于嘈杂，请使用 crepe 处理 f0
>
> 2. 如果省略 f0_predictor 参数，默认值为 rmvpe

**若需要浅扩散功能（可选），需要增加--use_diff 参数，比如:**

```bash
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

执行完以上步骤后生成的 dataset 目录便是预处理完成的数据，此时你可以按需删除 dataset_raw 文件夹了。

## 2.5 训练

### 2.5.1 扩散模型（可选）

So-VITS-SVC 4.1 的一个重大更新就是引入了浅扩散 (Shallow Diffusion) 机制，将 SoVITS 的原始输出音频转换为 Mel 谱图，加入噪声并进行浅扩散处理后经过声码器输出音频。经过测试，**原始输出音频在经过浅扩散处理后可以显著改善电音、底噪等问题，输出质量得到大幅增强**。

尚若需要浅扩散功能，需要训练扩散模型，训练前请确保你已经下载并正确放置好了 `NSF-HIFIGAN` （**参考 [2.2.3](#223-可选项-根据情况选择)**）,并且预处理生成 hubert 与 f0 时添加了 `--use_diff` 参数（**参考 [2.4.3](#243-生成-hubert-与-f0)**）

扩散模型训练方法为:

```bash
python train_diff.py -c configs/diffusion.yaml
```

### 2.5.2 主模型训练（必须）

```bash
python train.py -c configs/config.json -m 44k
```

模型训练结束后，模型文件保存在`logs/44k`目录下，扩散模型在`logs/44k/diffusion`下。

> [!NOTE]
>
> **模型怎样才算训练好了**？
>
> 1. 这是一个非常无聊且没有意义的问题。就好比上来就问老师我家孩子怎么才能学习好，除了你自己，没有人能回答这个问题。
> 2. 模型的训练关联于你的数据集质量、时长，所选的编码器、f0 算法，甚至一些超自然的玄学因素，即便你有一个成品模型，最终的转换效果也要取决于你的输入源以及推理参数。这不是一个线性的的过程，之间的变量实在是太多，所以你非得问“为什么我的模型出来不像啊”、“模型怎样才算训练好了”这样的问题，我只能说 WHO F\*\*KING KNOWS?
> 3. 但也不是一点办法没有，只能烧香拜佛了。我不否认烧香拜佛当然是一个有效的手段，但你也可以借助一些科学的工具，例如 Tensorboard 等，下方 2.5.3 就将教你怎么通过看 Tensorboard 来辅助了解训练状态，**当然，最强的辅助工具其实长在你自己身上，一个声学模型怎样才算训练好了? 塞上耳机，让你的耳朵告诉你吧**。
>
> **Epoch 和 Step 的关系**：
>
> 训练过程中会根据你在 `config.json` 中设置的保存步数（默认为 800 步，与 `eval_interval` 的值对应）保存一次模型。
> 请严格区分轮数 (Epoch) 和步数 (Step)：1 个 Epoch 代表训练集中的所有样本都参与了一次学习，1 Step 代表进行了一步学习，由于 batch_size 的存在，每步学习可以含有数条样本，因此，Epoch 和 Step 的换算如下：
> $Epoch = \frac{Step}{(\text{数据集条数}{\div}batch\_size)}$
> 训练默认 10000 轮后结束，但正常训练通常只需要数百轮即可有较好的效果。当你觉得训练差不多完成了，可以在训练终端按 Ctrl + C 中断训练。中断后只要没有重新预处理训练集，就可以**从最近一次保存点继续训练**。

### 2.5.3 Tensorboard

你可以用 Tensorboard 来查看训练过程中的损失函数值 (loss) 趋势，试听音频，从而辅助判断模型训练状态。**但是，就 So-VITS-SVC 这个项目而言，损失函数值(loss)并没有什么实际参考意义（你不用刻意对比研究这个值的高低），真正有参考意义的还是推理后靠你的耳朵来听！**

- 使用下面的命令打开 Tensorboard：

```bash
tensorboard --logdir=logs/44k
```

Tensorboard 是根据训练时默认每 200 步的评估生成日志的，如果训练未满 200 步，则 Tensorboard 中不会出现任何图像。200 这个数值可以通过修改 `config.json` 中的 `log_interval` 值来修改。

- Losses 详解

你不需要理解每一个 loss 的具体含义，大致来说：

- loss/g/f0、loss/g/mel 和 loss/g/total 应当是震荡下降的，并最终收敛在某个值
- loss/g/kl 应当是低位震荡的
- loss/g/fm 应当在训练的中期持续上升，并在后期放缓上升趋势甚至开始下降

观察 losses 曲线的趋势可以帮助你判断模型的训练状态。但 losses 并不能作为判断模型训练状态的唯一参考，**甚至它的参考价值其实并不大，你仍需要通过自己的耳朵来判断模型是否训练好了**。

> [!WARNING]
>
> 1. 对于小数据集（30 分钟甚至更小），在加载底模的情况下，不建议训练过久，这样是为了尽可能利用底模的优势。数千步甚至数百步就能有最好的结果。
> 2. Tensorboard 中的试听音频是根据你的验证集生成的，**无法代表模型最终的表现**。

# 3. 推理

✨ 推理时请先准备好需要推理的干声，确保干声无底噪/无混响/质量较好。你可以使用 [UVR5](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.6) 进行处理,得到干声。此外，我也制作了一个 [UVR5 人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/)

## 3.1 命令行推理

使用 inference_main.py 进行推理

```bash
# 例
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "你的推理音频.wav" -t 0 -s "speaker"
```

**必填项部分：**

- `-m` | `--model_path`：模型路径
- `-c` | `--config_path`：配置文件路径
- `-n` | `--clean_names`：wav 文件名列表，放在 raw 文件夹下
- `-t` | `--trans`：音高调整，支持正负（半音）
- `-s` | `--spk_list`：合成目标说话人名称
- `-cl` | `--clip`：音频强制切片，默认 0 为自动切片，单位为秒/s。

> [!NOTE]
>
> **音频切片**
>
> - 推理时，切片工具会将上传的音频根据静音段切分为数个小段，分别推理后合并为完整音频。这样做的好处是**小段音频推理显存占用低，因而可以将长音频切分推理以免爆显存**。切片阈值参数控制的是最小满刻度分贝值，低于这个值将被切片工具视为静音并去除。因此，当上传的音频比较嘈杂时，可以将该参数设置得高一些（如 -30），反之，可以将该值设置得小一些（如 -50）避免切除呼吸声和细小人声。
>
> - 开发团队近期的一项测试表明，较小的切片阈值（如-50）会改善输出的咬字，至于原理暂不清楚。
>
> **强制切片** `-cl` | `--clip`
>
> - 推理时，切片工具会将上传的音频根据静音段切分为数个小段，分别推理后合并为完整音频。但有时当人声过于连续，长时间不存在静音段时，切片工具也会相应切出来过长的音频，容易导致爆显存。自动音频切片功能则是设定了一个最长音频切片时长，初次切片后，如果存在长于该时长的音频切片，将会被按照该时长二次强制切分，避免了爆显存的问题。
> - 强制切片可能会导致音频从一个字的中间切开，分别推理再合并时可能会存在人声不连贯。你需要在高级设置中设置强制切片的交叉淡入长度来避免这一问题。

**可选项部分：部分具体见下一节**

- `-lg` | `--linear_gradient`：两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值 0，单位为秒
- `-f0p` | `--f0_predictor`：选择 F0 预测器，可选择 crepe,pm,dio,harvest,rmvpe,fcpe, 默认为 pm（注意：crepe 为原 F0 使用均值滤波器），不同 F0 预测器的优缺点请 **参考 [2.4.3 中的 F0 预测器的优缺点](#各个-f0-预测器的优缺点)**
- `-a` | `--auto_predict_f0`：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
- `-cm` | `--cluster_model_path`：聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填
- `-cr` | `--cluster_infer_ratio`：聚类方案或特征检索占比，范围 0-1，若没有训练聚类模型或特征检索则默认 0 即可
- `-eh` | `--enhance`：是否使用 NSF_HIFIGAN 增强器，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
- `-shd` | `--shallow_diffusion`：是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN 增强器将会被禁止
- `-usm` | `--use_spk_mix`：是否使用角色融合/动态声线融合
- `-lea` | `--loudness_envelope_adjustment`：输入源响度包络替换输出响度包络融合比例，越靠近 1 越使用输出响度包络
- `-fr` | `--feature_retrieval`：是否使用特征检索，如果使用聚类模型将被禁用，且 cm 与 cr 参数将会变成特征检索的索引路径与混合比例

> [!NOTE]
>
> **聚类模型/特征检索混合比例** `-cr` | `--cluster_infer_ratio`
>
> - 该参数控制的是使用聚类模型/特征检索模型时线性参与的占比。聚类模型和特征检索均可以有限提升音色相似度，但带来的代价是会降低咬字准确度（特征检索的咬字比聚类稍好一些）。该参数的范围为 0-1, 0 为不启用，越靠近 1, 则音色越相似，咬字越模糊。
> - 聚类模型和特征检索共用这一参数，当加载模型时使用了何种模型，则该参数控制何种模型的混合比例。
> - **注意，当未加载聚类模型或特征检索模型时，请保持改参数为 0，否则会报错。**

**浅扩散设置：**

- `-dm` | `--diffusion_model_path`：扩散模型路径
- `-dc` | `--diffusion_config_path`：扩散模型配置文件路径
- `-ks` | `--k_step`：扩散步数，越大越接近扩散模型的结果，默认 100
- `-od` | `--only_diffusion`：纯扩散模式，该模式不会加载 sovits 模型，以扩散模型推理
- `-se` | `--second_encoding`：二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差

> [!NOTE]
>
> **关于浅扩散步数** `-ks` | `--k_step`
>
> 完整的高斯扩散为 1000 步，当浅扩散步数达到 1000 步时，此时的输出结果完全是扩散模型的输出结果，So-VITS 模型将被抑制。浅扩散步数越高，越接近扩散模型输出的结果。**如果你只是想用浅扩散去除电音底噪，尽可能保留 So-VITS 模型的音色，浅扩散步数可以设定为 30-50**

> [!WARNING]
>
> 如果使用 `whisper-ppg` 声音编码器进行推理，需要将 `--clip` 设置为 25，`-lg` 设置为 1。否则将无法正常推理。

## 3.2 webUI 推理

使用以下命令打开 webui 界面，**上传模型并且加载，按照说明按需填写推理，上传推理音频，开始推理。**

参数具体的推理参数的详解和上面的 [命令行推理](#31-命令行推理) 参数一致，只不过搬到了交互式界面上去，并且附有简单的说明。

```bash
python webUI.py
```

> [!WARNING]
>
> **请务必查看 [命令行推理](#31-命令行推理) 以了解具体参数的含义。并且请特别注意 NOTE 和 WARNING 中的提醒！**

webUI 中还内置了 **文本转语音** 功能：

- 文本转语音使用微软的 edge_TTS 服务生成一段原始语音，再通过 So-VITS 将这段语音的声线转换为目标声线。So-VITS 只能实现歌声转换 (SVC) 功能，没有任何 **原生** 的文本转语音 (TTS) 功能！由于微软的 edge_TTS 生成的语音较为僵硬，没有感情，所有转换出来的音频当然也会这一。**如果你需要有感情的 TTS 功能，请移步 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 项目。**
- 目前文本转语音共支持 55 种语言，涵盖了大部分常见语言。程序会根据文本框内输入的文本自动识别语言并转换。
- 自动识别只能识别到语种，而某些语种可能涵盖不同的口音，说话人，如果使用了自动识别，程序会从符合该语种和指定性别的说话人种随机挑选一个来转换。如果你的目标语种说话人口音比较多（例如英语），建议手动指定一个口音的说话人。如果手动指定了说话人，则先前手动选择的性别会被抑制。

# 4. 增强效果的可选项

✨ 如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用(这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显)

## 4.1 自动 f0 预测

模型训练过程会训练一个 f0 预测器，是一个自动变调的功能，可以将模型音高匹配到推理源音高，用于说话声音转换时可以打开，能够更好匹配音调。**但转换歌声时请不要启用此功能！！！会严重跑调！！**

- 命令行推理：在 `inference_main` 中设置 `auto_predict_f0` 为 `true` 即可
- webUI 推理：勾选相应选项即可

## 4.2 聚类音色泄漏控制

聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点。

使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低。

- 训练方法：

```bash
# 使用CPU训练：
python cluster/train_cluster.py
# 或者使用GPU训练：
python cluster/train_cluster.py --gpu
```

训练结束后，模型的输出会在 `logs/44k/kmeans_10000.pt`

- 命令行推理过程：
  - `inference_main.py` 中指定 `cluster_model_path`
  - `inference_main.py` 中指定 `cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可
- webUI 推理过程：
  - 上传并加载聚类模型
  - 设置聚类模型/特征检索混合比例，0-1 之间，0 即不启用聚类/特征检索。使用聚类/特征检索能提升音色相似度，但会导致咬字下降（如果使用建议 0.5 左右）

## 4.3 特征检索

跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比。

- 训练过程：需要在生成 hubert 与 f0 后执行：

```bash
python train_index.py -c configs/config.json
```

训练结束后模型的输出会在 `logs/44k/feature_and_index.pkl`

- 命令行推理过程：
  - 需要首先制定 `--feature_retrieval`，此时聚类方案会自动切换到特征检索方案
  - `inference_main.py` 中指定 `cluster_model_path` 为模型输出文件
  - `inference_main.py` 中指定 `cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可
- webUI 推理过程：
  - 上传并加载聚类模型
  - 设置聚类模型/特征检索混合比例，0-1 之间，0 即不启用聚类/特征检索。使用聚类/特征检索能提升音色相似度，但会导致咬字下降（如果使用建议 0.5 左右）

## 4.4 声码器微调

在 So-VITS 中使用扩散模型时，经过扩散模型增强的 Mel 谱图会经过声码器（Vocoder）输出为最终音频。声码器在其中对输出音频的音质起到了决定性的作用。So-VITS-SVC 目前使用的是 [NSF-HiFiGAN 社区声码器](https://openvpi.github.io/vocoders/)，**实际上，你也可以用你自己的数据集对该声码器模型进行微调训练，在 So-VITS 的扩散流程中使用微调后的声码器，使其更符合你的模型任务。**

[SingingVocoders](https://github.com/openvpi/SingingVocoders) 项目提供了对声码器的微调方法，在 Diffusion-SVC 项目中，**使用微调声码器可以使输出音质得到大幅增强**。你也可以自行使用自己的数据集训练一个微调声码器，并在本整合包中使用。

1. 使用 [SingingVocoders](https://github.com/openvpi/SingingVocoders) 训练一个微调声码器，并获得其模型和配置文件
2. 将模型和配置文件放置在 `pretrain/{微调声码器名称}/` 下
3. 在推理加载模型时选择该模型对应的微调声码器

> [!WARNING]
>
> **目前仅支持微调的 NSF-HiFiGAN 声码器**

## 4.5 各模型保存的目录

截止上文，一共能够训练的 4 种模型都已经讲完了，使用下表总结一下这四种模型和配置文件。

webUI 中除了能够上传模型进行加载以外，也可以读取本地模型文件。你只需将下表这些模型先放入到一个文件夹内，再将该文件夹放到 trained 文件夹内，点击“刷新本地模型列表”，即可被 webUI 识别到。然后手动选择需要加载的模型进行加载即可。

**注意**：本地模型自动加载可能无法正常加载下表中的（可选）模型。

| 文件                     | 后缀    | 存放位置             |
| ------------------------ | ------- | -------------------- |
| So-VITS 模型             | `.pth`  | `logs/44k`           |
| So-VITS 模型配置文件     | `.json` | `configs`            |
| 扩散模型（可选）         | `.pt`   | `logs/44k/diffusion` |
| 扩散模型配置文件（可选） | `.yaml` | `configs`            |
| Kmeans 聚类模型（可选）  | `.pt`   | `logs/44k`           |
| 特征索引模型（可选）     | `.pkl`  | `logs/44k`           |

# 5.其他可选功能

✨ 此部分相较于前面的其他部分，重要性更低。除了 [5.1 模型压缩](#32-webui-推理) 是一个较为方便的功能以外，其余可选功能用到的概率较低，故此处仅参考官方文档并加以简单描述。

## 5.1 模型压缩

生成的模型含有继续训练所需的信息。如果**确认不再训练**，可以移除模型中此部分信息，得到约 1/3 大小的最终模型。

使用 compress_model.py

```bash
# 例如，我想压缩一个在logs/44k/目录下名字为G_30400.pth的模型，并且配置文件为configs/config.json，可以运行如下命令
python compress_model.py -c="configs/config.json" -i="logs/44k/G_30400.pth" -o="logs/44k/release.pth"
# 压缩后的模型保存在logs/44k/release.pth
```

> [!WARNING]
>
> **注意：压缩后的模型无法继续训练！**

## 5.2 声线混合

### 5.2.1 静态声线混合

**参考 `webUI.py` 文件中，小工具/实验室特性的静态声线融合。**

该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线。

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

**参考 `spkmix.py` 文件中关于动态声线混合的介绍**

角色混合轨道编写规则：

- 角色 ID : \[\[起始时间 1, 终止时间 1, 起始数值 1, 起始数值 1], [起始时间 2, 终止时间 2, 起始数值 2, 起始数值 2]]
- 起始时间和前一个的终止时间必须相同，第一个起始时间必须为 0，最后一个终止时间必须为 1 （时间的范围为 0-1）
- 全部角色必须填写，不使用的角色填\[\[0., 1., 0., 0.]]即可
- 融合数值可以随便填，在指定的时间段内从起始数值线性变化为终止数值，内部会自动确保线性组合为 1（凸组合条件），可以放心使用

命令行推理的时候使用 `--use_spk_mix` 参数即可启用动态声线混合。webUI 推理时勾选“动态声线融合”选项框即可。

## 5.3 Onnx 导出

使用 onnx_export.py。目前 onnx 模型只有 [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio) 需要使用到。更详细的操作和使用方法请移步 [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio) 仓库说明。

- 新建文件夹：`checkpoints` 并打开
- 在`checkpoints`文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如`aziplayer`
- 将你的模型更名为`model.pth`，配置文件更名为`config.json`，并放置到刚才创建的`aziplayer`文件夹下
- 将 onnx_export.py 中`path = "NyaruTaffy"` 的 `"NyaruTaffy"` 修改为你的项目名称，`path = "aziplayer" (onnx_export_speaker_mix，为支持角色混合的onnx导出)`
- 运行 `python onnx_export.py`
- 等待执行完毕，在你的项目文件夹下会生成一个`model.onnx`，即为导出的模型

注意：Hubert Onnx 模型请使用 [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio) 提供的模型，目前无法自行导出（fairseq 中 Hubert 有不少 onnx 不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出 shape 和结果都有问题）

# 6. 简单混音处理及成品导出

### 使用音频宿主软件处理推理后音频，具体流程比较麻烦，请参考 [配套视频教程](https://www.bilibili.com/video/BV1Hr4y197Cy/) | [UVR5 人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/) 或其他更专业的混音教程

# 附录：常见报错的解决办法

✨ **部分报错及解决方法，来自 [羽毛布団](https://space.bilibili.com/3493141443250876)大佬的 [相关专栏](https://www.bilibili.com/read/cv22206231) | [相关文档](https://www.yuque.com/umoubuton/ueupp5/ieinf8qmpzswpsvr)**

## 关于爆显存

如果你在终端或 WebUI 界面的报错中出现了这样的报错:

```bash
OutOfMemoryError: CUDA out of memory.Tried to allocate XX GiB (GPU O: XX GiB total capacity; XX GiB already allocated; XX MiB Free: XX GiB reserved in total by PyTorch)
```

不要怀疑，你的显卡显存或虚拟内存不够用了。以下是 100%解决问题的解决方法，照着做必能解决。请不要再在各种地方提问这个问题了

1. 在报错中找到 XX GiB already allocated 之后，是否显示 0 bytes free，如果是 0 bytes free 那么看第 2，3，4 步，如果显示 XX MiB free 或者 XX GiB free，看第 5 步
2. 如果是预处理的时候爆显存:
   - 换用对显存占用友好的 f0 预测器 (友好度从高到低: pm >= harvest >= rmvpe ≈ fcpe >> crepe)，建议首选 rmvpe 或 fcpe
   - 多进程预处理改为 1
3. 如果是训练的时候爆显存
   - 检查数据集有没有过长的切片（20 秒以上）
   - 调小批量大小 (batch size)
   - 更换一个占用低的项目
   - 去 AutoDL 等云算力平台上面租一张大显存的显卡跑
4. 如果是推理的时候爆显存:
   - 推理源 (千声) 不干净 (有残留的混响，伴奏，和声)，导致自动切片切不开。提取干声最佳实践请参考 [UVR5 歌曲人声分离教程](https://www.bilibili.com/video/BV1F4421c7qU/)
   - 调大切片闽值 (比如-40 调成-30，再大就不建议了，你也不想唱一半就被切一刀吧)
   - 设置强制切片，从 60 秒开始尝试，每次减小 10 秒，直到能成功推理
   - 使用 cpu 推理，速度会很慢但是不会爆显存
5. 如果显示仍然有空余显存却还是爆显存了，是你的虚拟内存不够大，调整到至少 50G 以上

## 安装依赖时出现的相关报错

**1. 安装 CUDA=11.7 的 Pytorch 时报错**

```
ERROR: Package 'networkx' requires a different Python: 3.8.9 not in '>=3.9
```

解决方法有两种：

- 升级 python 至 3.9（但可能造成不稳定）
- 保持 python 版本不变，先 `pip install networkx==3.0` 之后再进行 Pytorch 的安装。

**2. 依赖找不到导致的无法安装**

出现**类似**以下报错时：

```bash
ERROR: Could not find a version that satisfies the requirement librosa==0.9.1 (from versions: none)
ERROR: No matching distribution found for librosa==0.9.1
# 报错的主要特征是
No matching distribution found for xxxxx
Could not find a version that satisfies the requirement xxxx
```

具体解决方法为：更换安装源。手动安装这一依赖时添加下载源，以下是两个常用的镜像源地址

- 清华大学：<https://pypi.tuna.tsinghua.edu.cn/simple>
- 阿里云：<http://mirrors.aliyun.com/pypi/simple>

使用 `pip install [包名称] -i [下载源地址]` ，例如我想在阿里源下载 librosa 这个依赖，并且要求依赖版本是 0.9.1，那么应该在 cmd 中输入以下命令：

```bash
pip install librosa==0.9.1 -i http://mirrors.aliyun.com/pypi/simple
```

## 数据集预处理和模型训练时的相关报错

**1. 报错：`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0 in position xx`**

- 数据集文件名中不要包含中文或日文等非西文字符，特别注意**中文**括号，逗号，冒号，分号，引号等等都是不行的。改完名字**一定要**重新预处理，然后再进行训练！！！

**2. 报错：`The expand size of the tensor (768) must match the existing size (256) at non-singleton dimension 0.`**

- 把 dataset/44k 下的内容全部删了，重新走一遍预处理流程

**3. 报错：RuntimeError: DataLoader worker (pid(s) 13920) exited unexpectedly**

```bash
raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
RuntimeError: DataLoader worker (pid(s) 13920) exited unexpectedly
```

- 调小 batchsize 值，调大虚拟内存，重启电脑清理显存，直到 batchsize 值和虚拟内存合适不报错为止

**4. 报错：`torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with exit code 3221225477`**

- 调大虚拟内存，调小 batchsize 值，直到 batchsize 值和虚拟内存合适不报错为止

**5. 报错：`AssertionError: CPU training is not allowed.`**

- 没有解决方法：非 N 卡跑不了。（也不是完全跑不了，但如果你是纯萌新的话，那我的回答确实就是：跑不了）

**4. 报错：页面文件太小，无法完成操作。**

- 调大虚拟内存，具体的方法各种地方一搜就能搜到，不展开了

## 使用 WebUI 时相关报错

**1. webUI 启动或加载模型时**：

- 启动 webUI 时报错：`ImportError: cannot import name 'Schema' from 'pydantic'`
- webUI 加载模型时报错：`AttributeError("'Dropdown' object has no attribute 'update'")`
- **凡是报错中涉及到 fastapi, gradio, pydantic 这三个依赖的报错**

**解决方法**：

- 需限制部分依赖版本，在安装完 `requirements_win.txt` 后，在 cmd 中依次输入以下命令以更新依赖包：

```bash
pip install --upgrade fastapi==0.84.0
pip install --upgrade gradio==3.41.2
pip install --upgrade pydantic==1.10.12
```

**2. 报错：`Given groups=1, weight of size [xxx, 256, xxx], expected input[xxx, 768, xxx] to have 256 channels, but got 768 channels instead`**
或 **报错: 配置文件中的编码器与模型维度不匹配**

- 原因：v1 分支的模型用了 vec768 的配置文件，如果上面报错的 256 的 768 位置反过来了那就是 vec768 的模型用了 v1 的配置文件。
- 解决方法：检查配置文件中的 `ssl_dim` 一项，如果这项是 256，那你的 `speech_encoder` 应当修改为 `vec256|9`，如果是 768，则是 `vec768|12`

**3. 报错：`'HParams' object has no attribute 'xxx'`**

- 无法找到音色，一般是配置文件和模型没对应，打开配置文件拉到最下面看看有没有你训练的音色

# 感谢名单

- so-vits-svc | [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
- GPT-SoVITS | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- SingingVocoders | [SingingVocoders](https://github.com/openvpi/SingingVocoders)
- MoeVoiceStudio | [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio)
- up 主 [inifnite_loop](https://space.bilibili.com/286311429) | [相关视频](https://www.bilibili.com/video/BV1Bd4y1W7BN) | [相关专栏](https://www.bilibili.com/read/cv21425662)
- up 主 [羽毛布団](https://space.bilibili.com/3493141443250876) | [一些报错的解决办法](https://www.bilibili.com/read/cv22206231) | [常见报错解决方法](https://www.yuque.com/umoubuton/ueupp5/ieinf8qmpzswpsvr)
- 所有提供训练音频样本的人员
- 您
