本帮助文档为项目 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 的详细中文安装、调试、推理教程，您也可以直接选择官方[README](https://github.com/svc-develop-team/so-vits-svc#readme)文档
撰写：Sucial [点击跳转B站主页](https://space.bilibili.com/445022409)
# 写在开头：与3.0版本相比，4.0和4.1版本的安装、训练、推理操作更为简单
# 建议直接点击访问[官方文档](https://github.com/svc-develop-team/so-vits-svc)
----
# 2023.8.2文档更新：
## 1. 提供4.1训练底模，需自行下载，下载地址：https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model 还包含扩散模型训练底模

## 2. 提供4.0训练底模，需自行下载，下载地址：https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/tree/main/sovits_768l12_pre_large_320k 并需要改名为G_0.pth和D_0.pth

## 3. 提供3.0训练底模，需自行下载，下载地址：https://pan.baidu.com/s/1uw6W3gOBvMbVey1qt_AzhA?pwd=80eo 提取码：80eo 

## 4. 修改了一下文档内容。

# 其实到这里你完全可以参考官方的文档来一步一步配置了，但如果你不清楚前置环境配置，可以继续往下阅读下面文章的第一部分 1. 环境依赖 即可
----
# 下面的文章仅介绍4.0版本的安装方法（其实是懒的更新）因为4.1的安装过程官方写的真的很详细！！！

# ✅SoftVC VITS Singing Voice Conversion 教程目录

## 参考资料

我写的教程文档：https://github.com/SUC-DriverOld/so-vits-svc-Chinese-Detaild-Documents

官方README文档：https://github.com/svc-develop-team/so-vits-svc

一些报错的解决办法（来自B站up：**羽毛布団**）：https://www.bilibili.com/read/cv22206231

## 0. 用前须知

- 法律依据
- 硬件需求
- 提前准备
- 训练周期

## 1. 环境依赖

涉及软件：Cuda，Python，FFmpeg

- Cuda
- Python
- Pytorch
- 依赖库
- FFmpeg

## 2. 配置及训练

涉及软件：slicer-gui，Audition

- 预下载模型及预下载底模
- 训练集准备
  - 数据集准备
  - 数据集预处理
- 训练
  - 主模型训练
  - 扩散训练（可选）
  - 使用Tensorboard跟进训练进度及收敛判断

## 3. 推理

涉及软件：Ultimate Vocal Remover

- 命令行推理
- WebUI推理

## 4. 增强效果的可选项

- 自动f0预测
- 聚类音色泄漏控制
- 特征检索

## 5. 其他可选项

- 模型压缩
- 声线混合（本教程不讲）
  - 静态声线混合（本教程不讲）
  - 动态声线混合（本教程不讲）
- Onnx导出（本教程不讲）

## 6. 简单混音处理及成品导出

- 以FL Studio或Studio One为例

## 附录：常见报错的解决办法



# SoftVC VITS Singing Voice Conversion 教程

# ✅0. 用前须知

## 0.0 任何国家，地区，组织和个人使用此项目必须遵守以下法律

#### 《民法典》

##### 第一千零一十九条

任何组织或者个人**不得**以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。**未经**肖像权人同意，**不得**制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。**未经**肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。
**对自然人声音的保护，参照适用肖像权保护的有关规定**

##### 第一千零二十四条

【名誉权】民事主体享有名誉权。任何组织或者个人**不得**以侮辱、诽谤等方式侵害他人的名誉权。

##### 第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》

#### 《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=中华人民共和国刑法)》

#### 《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 本教程仅供交流与学习使用，请勿用于违法违规或违反公序良德等不良用途 

#### 出于对音源提供者的尊重请勿用于鬼畜用途 

### 0.0.1. 继续使用视为已同意本教程所述相关条例，本教程已进行劝导义务，不对后续可能存在问题负责。 

1. 本教程内容**仅代表个人**，均不代表so-vits-svc团队及原作者观点
2. 本教程涉及到的开源代码请自行**遵守其开源协议**
3. 本教程默认使用由**so-vits-svc团队维护**的仓库
4. 若制作视频发布，**推荐注明**使用项目的**Github**链接，tag**推荐**使用**so-vits-svc**以便和其他基于技术进行区分
5. 云端训练和推理部分可能涉及资金使用，如果你是**未成年人**，请在**获得监护人的许可与理解后**进行，未经许可引起的后续问题，本教程**概不负责**
6. 本地训练（尤其是在硬件较差的情况下）可能需要设备长时间**高负荷**运行，请做好设备养护和散热措施
7. 请确保你制作数据集的数据来源**合法合规**，且数据提供者明确你在制作什么以及可能造成的后果
8. 出于设备原因，本教程仅在**Windows**系统下进行过测试，Mac和Linux请确保自己有一定解决问题能力
9. 该项目为**歌声合成**项目，**无法**进行其他用途，请知悉

### 0.0.2. 声明

本项目为开源、离线的项目，SvcDevelopTeam的所有成员与本项目的所有开发者以及维护者（以下简称贡献者）对本项目没有控制力。本项目的贡献者从未向任何组织或个人提供包括但不限于数据集提取、数据集加工、算力支持、训练支持、推理等一切形式的帮助；本项目的贡献者不知晓也无法知晓使用者使用该项目的用途。故一切基于本项目训练的AI模型和合成的音频都与本项目贡献者无关。一切由此造成的问题由使用者自行承担。

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

1. 推理目前分为**命令行推理**和**WebUI推理**，对速度要求不高的话CPU和GPU均可使用
2. 至少需要**6G以上**显存的**NVIDIA显卡**（如RTX3060）
3. 云端一般常见的为V100（16G）、V100（32G）、A100（40G）、A100（80G）等显卡，部分云端提供RTX3090等显卡

## 0.2 提前准备

1. **至少**准备200条8s（约30分钟**持续说话**时长，即约1.5小时**正常说话**采样）左右时长的**干净**人声（**无底噪，无混响**）作为训练集。并且最好保持说话者**情绪起伏波动较小**，人声**响度合适**，并且做好**响度匹配**
2. 请提前准备训练需要用到的**底模**（**挺重要的**）
3. **须知**：歌声作为训练集**只能**用来推理歌声，但语音作为训练集即可以推理歌声，也可以用来生成TTS。但用语音作为训练集可能使**高音和低音推理出现问题**（即缺少高低音训练样本），有一种可行的解决方法是模型融合。
4. 推理：需准备**底噪<30dB**，尽量**不要带过多混响和和声**的**干音**进行推理
5. **须知**：推理女声歌曲时，建议用女声训练模型，同理男声也类似

## 0.3 训练周期

在**有底模**的前提下，选取**200条音频**作为训练集，经多次测试（RTX3060, `batch_size = 3`）得到以下结论：

1. 模型达到基本收敛的训练步数10w+（若每晚训练约8小时，需要约7天+）
2. 模型大概能用（一些高低音可能有问题）的训练步数约2w-3w（若每晚训练约8小时，需要约2-3天）
3. 模型基本能用（没大问题）的训练步数约5w-8w（若每晚训练约8小时，需要约4-5天）



# ✅1. 环境依赖

> - **本项目需要的环境：**
>   NVIDIA-CUDA
>   Python = 3.8.9
>   Pytorch
>   FFmpeg

## 1.1 Cuda

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

- 目前（2023/3/28）为止pytorch最高支持到```cuda11.7```
- 如果您在上述第一步中查看到自己的Cuda版本>11.7，请依然选择11.7进行下载安装（Cuda有版本兼容性）并且安装完成后再次在cmd输入```nvidia-smi.exe```并不会出现cuda版本变化，即任然显示的是>11,7的版本
- **Cuda的卸载方法：**打开控制面板-程序-卸载程序，将带有```NVIDIA CUDA```的程序全部卸载即可（一共5个）

## 1.2 Python

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

## 1.3 Pytorch

- 首先我们需要**单独安装**```torch```, ```torchaudio```, ```torchvision```这三个库，下面提供两种方法

#### 方法1（便捷）

> 直接前往 [Pytorch官网](https://pytorch.org/get-started/locally/) 选择所需版本然后复制Run this Command栏显示的命令至cmd安装（不建议）

#### 方法2（较慢但稳定）

- **前往该地址使用```Ctrl+F```搜索直接下载whl包** [点击前往](https://download.pytorch.org/whl/)
  
  > - 这个项目需要的是
  >   ```torch==1.10.0+cu113```
  >   ```torchaudio==0.10.0+cu113```
  >   ```1.10.0```和   ```0.10.0```表示是```pytorch```版本，```cu113```表示```cuda```版本```11.3```
  >   以此类推，请选择**适合自己的版本**安装

- **下面我将以```Cuda11.7```版本为例**
  ***--示例开始--***
  
  > - 我们需要安装以下三个库
  > 1. [torch-1.13.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-win_amd64.whl) 其中cp310指```python3.10```, ```win-amd64```表示windows 64位操作系统
  > 2. [torchaudio-0.13.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torchaudio-0.13.0%2Bcu117-cp310-cp310-win_amd64.whl)
  > 3. [torchvision-0.14.0+cu117-cp310-cp310-win_amd64.whl](https://download.pytorch.org/whl/cu117/torchvision-0.14.0%2Bcu117-cp310-cp310-win_amd64.whl)

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

## 1.4 安装依赖

- 在项目文件夹内右击空白处选择 **在终端中打开** 并执行下面命令以安装库（若出现报错请尝试用```pip install [库名称]```重新单独安装直至成功）

```shell
    pip install -r requirements.txt
```

## 1.5 FFmpeg

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

# ✅2. 配置及训练（参考官方文档）

## 2.0 关于兼容4.0模型的问题

+ 可通过修改4.0模型的config.json对4.0的模型进行支持，需要在config.json的model字段中添加speech_encoder字段，具体见下

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

配置及训练

## 2.2 预先下载的模型文件

#### **必须项**

**以下编码器需要选择一个使用**

##### **1. 若使用contentvec作为声音编码器（推荐）**

`vec768l12`与`vec256l9` 需要该编码器

+ contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  + 放在`pretrain`目录下

或者下载下面的ContentVec，大小只有199MB，但效果相同:

+ contentvec ：[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
  + 将文件名改为`checkpoint_best_legacy_500.pt`后，放在`pretrain`目录下

```shell
# contentvec
wget -P pretrain/ http://obs.cstcloud.cn/share/obs/sankagenkeshi/checkpoint_best_legacy_500.pt
# 也可手动下载放在pretrain目录
```

##### **2. 若使用hubertsoft作为声音编码器**

+ soft vc hubert：[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  + 放在`pretrain`目录下

##### **3. 若使用Whisper-ppg作为声音编码器**

+ 下载模型 [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 该模型适配`whisper-ppg`
+ 下载模型 [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), 该模型适配`whisper-ppg-large`
  + 放在`pretrain`目录下

##### **4. 若使用cnhubertlarge作为声音编码器**

+ 下载模型 [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)
  + 放在`pretrain`目录下

##### **5. 若使用dphubert作为声音编码器**

+ 下载模型 [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)
  + 放在`pretrain`目录下

##### **6. 若使用OnnxHubert/ContentVec作为声音编码器**

+ 下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
  + 放在`pretrain`目录下

#### **编码器列表**

- "vec768l12"

+ "vec256l9"
+ "vec256l9-onnx"
+ "vec256l12-onnx"
+ "vec768l9-onnx"
+ "vec768l12-onnx"
+ "hubertsoft-onnx"
+ "hubertsoft"
+ "whisper-ppg"
+ "cnhubertlarge"
+ "dphubert"
+ "whisper-ppg-large"

#### **可选项(强烈建议使用)**

+ 预训练底模文件： `G_0.pth` `D_0.pth`
  + 放在`logs/44k`目录下

+ 扩散模型预训练底模文件： `model_0.pt`
  + 放在`logs/44k/diffusion`目录下

从svc-develop-team(待定)或任何其他地方获取Sovits底模

扩散模型引用了[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)的Diffusion Model，底模与[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)的扩散模型底模通用，可以去[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)获取扩散模型的底模

虽然底模一般不会引起什么版权问题，但还是请注意一下，比如事先询问作者，又或者作者在模型描述中明确写明了可行的用途



> ### 提供4.1训练底模，需自行下载，下载地址：https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model 还包含扩散模型训练底模
>
> ### 提供4.0训练底模，需自行下载，下载地址：https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/tree/main/sovits_768l12_pre_large_320k 并需要改名为G_0.pth和D_0.pth
>
> ### 提供3.0训练底模，需自行下载，下载地址：https://pan.baidu.com/s/1uw6W3gOBvMbVey1qt_AzhA?pwd=80eo 提取码：80eo 



#### **可选项(根据情况选择)**

如果使用`NSF-HIFIGAN增强器`或`浅层扩散`的话，需要下载预训练的NSF-HIFIGAN模型，如果不需要可以不下载

+ 预训练的NSF-HIFIGAN声码器 ：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
  + 解压后，将四个文件放在`pretrain/nsf_hifigan`目录下

```shell
# nsf_hifigan
wget -P pretrain/ https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip -od pretrain/nsf_hifigan pretrain/nsf_hifigan_20221211.zip
# 也可手动下载放在pretrain/nsf_hifigan目录
# 地址：https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1
```

## 2.3 数据集准备

仅需要以以下文件结构将数据集放入dataset_raw目录即可

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

**如果你使用Whisper-ppg声音编码器进行训练，所有的切片长度必须小于30s**

### 2.4.1 重采样至44100Hz单声道

```shell
python resample.py
```

#### 注意

虽然本项目拥有重采样、转换单声道与响度匹配的脚本resample.py，但是默认的响度匹配是匹配到0db。这可能会造成音质的受损。而python的响度匹配包pyloudnorm无法对电平进行压限，这会导致爆音。所以建议可以考虑使用专业声音处理软件如`adobe audition`等软件做响度匹配处理。若已经使用其他软件做响度匹配，可以在运行上述命令时添加`--skip_loudnorm`跳过响度匹配步骤。如：

```shell
python resample.py --skip_loudnorm
```

### 2.4.2 自动划分训练集、验证集，以及自动生成配置文件

```shell
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder拥有七个选择

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
whisper-ppg-large
cnhubertlarge
dphubert
```

如果省略speech_encoder参数，默认值为vec768l12

**使用响度嵌入**

若使用响度嵌入，需要增加`--vol_aug`参数，比如：

```shell
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

使用后训练出的模型将匹配到输入源响度，否则为训练集响度。

#### 此时可以在生成的config.json与diffusion.yaml修改部分参数

+ `keep_ckpts`：训练时保留最后几个模型，`0`为保留所有，默认只保留最后`3`个

+ `all_in_mem`,`cache_all_data`：加载所有数据集到内存中，某些平台的硬盘IO过于低下、同时内存容量 **远大于** 数据集体积时可以启用

+ `batch_size`：单次训练加载到GPU的数据量，调整到低于显存容量的大小即可

+ `vocoder_name` : 选择一种声码器，默认为`nsf-hifigan`.

##### **声码器列表**

```
nsf-hifigan
nsf-snake-hifigan
```

### 2.4.3 生成hubert与f0

```shell
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor拥有四个选择

```
crepe
dio
pm
harvest
```

如果训练集过于嘈杂，请使用crepe处理f0

如果省略f0_predictor参数，默认值为dio

尚若需要浅扩散功能（可选），需要增加--use_diff参数，比如

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

使用 [inference_main.py](inference_main.py)

```shell
# 例
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

必填项部分：

+ `-m` | `--model_path`：模型路径
+ `-c` | `--config_path`：配置文件路径
+ `-n` | `--clean_names`：wav 文件名列表，放在 raw 文件夹下
+ `-t` | `--trans`：音高调整，支持正负（半音）
+ `-s` | `--spk_list`：合成目标说话人名称
+ `-cl` | `--clip`：音频强制切片，默认0为自动切片，单位为秒/s

可选项部分：部分具体见下一节

+ `-lg` | `--linear_gradient`：两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒
+ `-f0p` | `--f0_predictor`：选择F0预测器,可选择crepe,pm,dio,harvest,默认为pm(注意：crepe为原F0使用均值滤波器)
+ `-a` | `--auto_predict_f0`：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
+ `-cm` | `--cluster_model_path`：聚类模型或特征检索索引路径，如果没有训练聚类或特征检索则随便填
+ `-cr` | `--cluster_infer_ratio`：聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可
+ `-eh` | `--enhance`：是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
+ `-shd` | `--shallow_diffusion`：是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止
+ `-usm` | `--use_spk_mix`：是否使用角色融合/动态声线融合
+ `-lea` | `--loudness_envelope_adjustment`：输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络
+ `-fr` | `--feature_retrieval`：是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例

浅扩散设置：

+ `-dm` | `--diffusion_model_path`：扩散模型路径
+ `-dc` | `--diffusion_config_path`：扩散模型配置文件路径
+ `-ks` | `--k_step`：扩散步数，越大越接近扩散模型的结果，默认100
+ `-od` | `--only_diffusion`：纯扩散模式，该模式不会加载sovits模型，以扩散模型推理
+ `-se` | `--second_encoding`：二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差

### 注意

如果使用`whisper-ppg` 声音编码器进行推理，需要将`--clip`设置为25，`-lg`设置为1。否则将无法正常推理。

## 3.2 WebUI推理

使用以下命令打开webui界面，推理参数参考3.1

```shell
chcp 65001
@echo off
python webUI.py
pause
```





# ✅4. 增强效果的可选项

如果前面的效果已经满意，或者没看明白下面在讲啥，那后面的内容都可以忽略，不影响模型使用(这些可选项影响比较小，可能在某些特定数据上有点效果，但大部分情况似乎都感知不太明显)

### 自动f0预测

4.0模型训练过程会训练一个f0预测器，对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动的，但转换歌声时请不要启用此功能！！！会严重跑调！！

+ 在inference_main中设置auto_predict_f0为true即可

### 聚类音色泄漏控制

介绍：聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点

使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低

+ 训练过程：
  + 使用cpu性能较好的机器训练，据我的经验在腾讯云6核cpu训练每个speaker需要约4分钟即可完成训练
  + 执行`python cluster/train_cluster.py`，模型的输出会在`logs/44k/kmeans_10000.pt`
  + 聚类模型目前可以使用gpu进行训练，执行`python cluster/train_cluster.py --gpu`
+ 推理过程：
  + `inference_main.py`中指定`cluster_model_path`
  + `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可

### 特征检索

介绍：跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比，

+ 训练过程：
  首先需要在生成hubert与f0后执行：

```shell
python train_index.py -c configs/config.json
```

模型的输出会在`logs/44k/feature_and_index.pkl`

+ 推理过程：
  + 需要首先制定`--feature_retrieval`，此时聚类方案会自动切换到特征检索方案
  + `inference_main.py`中指定`cluster_model_path` 为模型输出文件
  + `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可



# ✅5.其他可选项

## 5.1 模型压缩

生成的模型含有继续训练所需的信息。如果确认不再训练，可以移除模型中此部分信息，得到约 1/3 大小的最终模型。

使用 [compress_model.py](compress_model.py)

```shell
# 例
python compress_model.py -c="configs/config.json" -i="logs/44k/G_30400.pth" -o="logs/44k/release.pth"
```

## 5.2 声线混合（本教程不讲）

### 5.2.1 静态声线混合

**参考`webUI.py`文件中，小工具/实验室特性的静态声线融合。**

介绍:该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线
**注意：**

1. 该功能仅支持单说话人的模型
2. 如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个SpaekerID下的声音
3. 保证所有待混合模型的config.json中的model字段是相同的
4. 输出的混合模型可以使用待合成模型的任意一个config.json，但聚类模型将不能使用
5. 批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
6. 混合比例调整建议大小在0-100之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
7. 混合完毕后，文件将会保存在项目根目录中，文件名为output.pth
8. 凸组合模式会将混合比例执行Softmax使混合比例相加为1，而线性组合模式不会

### 5.2.2 动态声线混合

**参考`spkmix.py`文件中关于动态声线混合的介绍**

角色混合轨道 编写规则：

角色ID : \[\[起始时间1, 终止时间1, 起始数值1, 起始数值1], [起始时间2, 终止时间2, 起始数值2, 起始数值2]]

起始时间和前一个的终止时间必须相同，第一个起始时间必须为0，最后一个终止时间必须为1 （时间的范围为0-1）

全部角色必须填写，不使用的角色填\[\[0., 1., 0., 0.]]即可

融合数值可以随便填，在指定的时间段内从起始数值线性变化为终止数值，内部会自动确保线性组合为1（凸组合条件），可以放心使用

推理的时候使用`--use_spk_mix`参数即可启用动态声线混合

## 5.3 Onnx导出（本教程不讲）

使用 [onnx_export.py](onnx_export.py)

+ 新建文件夹：`checkpoints` 并打开
+ 在`checkpoints`文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如`aziplayer`
+ 将你的模型更名为`model.pth`，配置文件更名为`config.json`，并放置到刚才创建的`aziplayer`文件夹下
+ 将 [onnx_export.py](onnx_export.py) 中`path = "NyaruTaffy"` 的 `"NyaruTaffy"` 修改为你的项目名称，`path = "aziplayer" (onnx_export_speaker_mix，为支持角色混合的onnx导出)`
+ 运行 [onnx_export.py](onnx_export.py)
+ 等待执行完毕，在你的项目文件夹下会生成一个`model.onnx`，即为导出的模型

注意：Hubert Onnx模型请使用MoeSS提供的模型，目前无法自行导出（fairseq中Hubert有不少onnx不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出shape和结果都有问题）



# ✅6. 简单混音处理及成品导出

### 以FL studio或Studio One为例



# ✅附录：常见报错的解决办法

## 报错及解决方法，来自https://www.bilibili.com/read/cv22206231



**报错：`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0 in position xx`**
答：数据集文件名中不要包含中文或日文等非西文字符。

**报错：页面文件太小，无法完成操作。**
答：调整一下虚拟内存大小，具体的方法各种地方一搜就能搜到，不展开了。

**报错：`UnboundLocalError: local variable 'audio' referenced before assignment`**
答：上传的推理音频需要是16位整数wav格式，用Au转换一下就好。或者装个ffmpeg一劳永逸地解决问题。

**报错：`AssertionError: CPU training is not allowed.`**
答：非N卡跑不了的。

**报错：`torch.cuda.OutOfMemoryError: CUDA out of memory`**
答：爆显存了，试着把batch_size改小，改到1还爆的话建议云端训练。

**报错：`RuntimeError: DataLoader worker (pid(s) xxxx) exited unexpectedly`**
答：把虚拟内存再调大一点。

**报错：`NotImplementedError: Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for no`**
答：数据集切片切太长了，5-10秒差不多。

**报错：`CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling 'cublasCreate(handle)'`**
答：爆显存了，基本上跟CUDA有关的报错大都是爆显存……

**报错：`torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with exit code 3221225477`**
答：调大虚拟内存，管理员运行脚本

**报错：`'HParams' object has no attribute 'xxx'`**
答：无法找到音色，一般是配置文件和模型没对应，打开配置文件拉到最下面看看有没有你训练的音色

**报错：`The expand size of the tensor (768) must match the existing size (256) at non-singleton dimension 0.`**
答：把dataset/44k下的内容全部删了，重新走一遍预处理流程

**报错：`Given groups=1, weight of size [xxx, 256, xxx], expected input[xxx, 768, xxx] to have 256 channels, but got 768 channels instead`**
答：v1分支的模型用了vec768的配置文件，如果上面报错的256的768位置反过来了那就是vec768的模型用了v1的配置文件

>
> - **以下是对本文档的撰写有帮助的感谢名单：**
> so-vits-svc [官方源代码和帮助文档](https://github.com/MaxMax2016/so-vits-svc)
> B站up主inifnite_loop [相关视频](https://www.bilibili.com/video/BV1Bd4y1W7BN) [相关专栏](https://www.bilibili.com/read/cv21425662)
> 所有提供训练音频样本的人员
