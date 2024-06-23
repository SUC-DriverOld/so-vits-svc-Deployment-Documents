<div align="center">

# SoftVC VITS Singing Voice Conversion Local Deployment Tutorial

English | [简体中文](README_zh_CN.md)

**Last Updated: June 22, 2024.**

This help document provides detailed installation, debugging, and inference tutorials for the project [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc). You can also directly refer to the official [README](https://github.com/svc-develop-team/so-vits-svc#readme) documentation.

Written by Sucial. [Bilibili](https://space.bilibili.com/445022409) | [Github](https://github.com/SUC-DriverOld)

</div>

---

✨ **I wrote a script for a one-click setup environment and launching webUI for so-vits-svc: [so-vits-svc-webUI-QuickStart-bat](https://github.com/SUC-DriverOld/so-vits-svc-webUI-QuickStart-bat). Welcome to use it!**

✨ **Click to view: [Accompanying Video Tutorial](https://www.bilibili.com/video/BV1Hr4y197Cy/) | [UVR5 Vocal Separation Tutorial](https://www.bilibili.com/video/BV1F4421c7qU/) (Note: The accompanying video may be outdated. Refer to the latest tutorial documentation for accurate information!)**

✨ **Related Resources: [Official README Documentation](https://github.com/svc-develop-team/so-vits-svc) | [Common Error Solutions](https://www.yuque.com/umoubuton/ueupp5/ieinf8qmpzswpsvr) | [羽毛布団](https://space.bilibili.com/3493141443250876)**

> [!IMPORTANT]
>
> 1. **Important! Read this first!** If you do not want to configure the environment manually or are looking for an integration package, please use the integration package by [羽毛布団](https://space.bilibili.com/3493141443250876).
> 2. **About old version tutorials**: For the so-vits-svc3.0 version tutorial, please switch to the [3.0 branch](https://github.com/SUC-DriverOld/so-vits-svc-Chinese-Detaild-Documents/tree/3.0). This branch is no longer being updated!
> 3. **Continuous improvement of the documentation**: If you encounter errors not mentioned in this document, you can ask questions in the issues section. For project bugs, please report issues to the original project. If you want to improve this tutorial, feel free to submit a PR!

# Tutorial Index

- [SoftVC VITS Singing Voice Conversion Local Deployment Tutorial](#softvc-vits-singing-voice-conversion-local-deployment-tutorial)
- [Tutorial Index](#tutorial-index)
- [0. Before You Use](#0-before-you-use)
  - [Any country, region, organization, or individual using this project must comply with the following laws:](#any-country-region-organization-or-individual-using-this-project-must-comply-with-the-following-laws)
    - [《民法典》](#民法典)
    - [第一千零一十九条](#第一千零一十九条)
    - [第一千零二十四条](#第一千零二十四条)
    - [第一千零二十七条](#第一千零二十七条)
    - [《中华人民共和国宪法》|《中华人民共和国刑法》|《中华人民共和国民法典》|《中华人民共和国合同法》](#中华人民共和国宪法中华人民共和国刑法中华人民共和国民法典中华人民共和国合同法)
  - [0.1 Usage Regulations](#01-usage-regulations)
  - [0.2 Hardware Requirements](#02-hardware-requirements)
  - [0.3 Preparation](#03-preparation)
- [1. Environment Dependencies](#1-environment-dependencies)
  - [1.1 so-vits-svc4.1 Source Code](#11-so-vits-svc41-source-code)
  - [1.2 Cuda](#12-cuda)
  - [1.3 Python](#13-python)
  - [1.4 Pytorch](#14-pytorch)
  - [1.5 Installation of Other Dependencies](#15-installation-of-other-dependencies)
  - [1.6 FFmpeg](#16-ffmpeg)
- [2. Configuration and Training](#2-configuration-and-training)
  - [2.1 Issues Regarding Compatibility with the 4.0 Model](#21-issues-regarding-compatibility-with-the-40-model)
  - [2.2 Pre-downloaded Model Files](#22-pre-downloaded-model-files)
    - [2.2.1 Mandatory Items](#221-mandatory-items)
      - [Detailed Explanation of Each Encoder](#detailed-explanation-of-each-encoder)
    - [2.2.2 Pre-trained Base Model (Strongly Recommended)](#222-pre-trained-base-model-strongly-recommended)
    - [2.2.3 Optional Items (Choose as Needed)](#223-optional-items-choose-as-needed)
  - [2.3 Data Preparation](#23-data-preparation)
  - [2.4 Data Preprocessing](#24-data-preprocessing)
    - [2.4.0 Audio Slicing](#240-audio-slicing)
    - [2.4.1 Resampling to 44100Hz Mono](#241-resampling-to-44100hz-mono)
    - [2.4.2 Automatic Dataset Splitting and Configuration File Generation](#242-automatic-dataset-splitting-and-configuration-file-generation)
      - [Using Loudness Embedding](#using-loudness-embedding)
    - [2.4.3 Modify Configuration Files as Needed](#243-modify-configuration-files-as-needed)
      - [config.json](#configjson)
      - [diffusion.yaml](#diffusionyaml)
    - [2.4.3 Generating Hubert and F0](#243-generating-hubert-and-f0)
      - [Pros and Cons of Each F0 Predictor](#pros-and-cons-of-each-f0-predictor)
  - [2.5 Training](#25-training)
    - [2.5.1 Main Model Training (Required)](#251-main-model-training-required)
    - [2.5.2 Diffusion Model (Optional)](#252-diffusion-model-optional)
    - [2.5.3 Tensorboard](#253-tensorboard)
- [3. Inference](#3-inference)
  - [3.1 Command-line Inference](#31-command-line-inference)
  - [3.2 webUI Inference](#32-webui-inference)
- [4. Optional Enhancements](#4-optional-enhancements)
  - [4.1 Automatic F0 Prediction](#41-automatic-f0-prediction)
  - [4.2 Clustering Timbre Leakage Control](#42-clustering-timbre-leakage-control)
  - [4.3 Feature Retrieval](#43-feature-retrieval)
  - [4.4 Vocoder Fine-tuning](#44-vocoder-fine-tuning)
  - [4.5 Directories for Saved Models](#45-directories-for-saved-models)
- [5. Other Optional Features](#5-other-optional-features)
  - [5.1 Model Compression](#51-model-compression)
  - [5.2 Voice Mixing](#52-voice-mixing)
    - [5.2.1 Static Voice Mixing](#521-static-voice-mixing)
    - [5.2.2 Dynamic Voice Mixing](#522-dynamic-voice-mixing)
  - [5.3 Onnx Export](#53-onnx-export)
- [6. Simple Mixing and Exporting Finished Product](#6-simple-mixing-and-exporting-finished-product)
  - [Use Audio Host Software to Process Inferred Audio](#use-audio-host-software-to-process-inferred-audio)
- [Appendix: Common Errors and Solutions](#appendix-common-errors-and-solutions)
  - [About Out of Memory (OOM)](#about-out-of-memory-oom)
  - [Common Errors and Solutions When Installing Dependencies](#common-errors-and-solutions-when-installing-dependencies)
  - [Common Errors During Dataset Preprocessing and Model Training](#common-errors-during-dataset-preprocessing-and-model-training)
  - [Errors When Using WebUI\*\*](#errors-when-using-webui)
- [Acknowledgements](#acknowledgements)

# 0. Before You Use

### Any country, region, organization, or individual using this project must comply with the following laws:

#### 《[民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 第一千零一十九条

任何组织或者个人**不得**以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。**未经**肖像权人同意，**不得**制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。**未经**肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。
**对自然人声音的保护，参照适用肖像权保护的有关规定**

#### 第一千零二十四条

【名誉权】民事主体享有名誉权。任何组织或者个人**不得**以侮辱、诽谤等方式侵害他人的名誉权。

#### 第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》|《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=中华人民共和国刑法)》|《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》|《[中华人民共和国合同法](http://www.npc.gov.cn/zgrdw/npc/lfzt/rlyw/2016-07/01/content_1992739.htm)》

## 0.1 Usage Regulations

> [!WARNING]
>
> 1. **This tutorial is for communication and learning purposes only. Do not use it for illegal activities, violations of public order, or other unethical purposes. Out of respect for the providers of audio sources, do not use this for inappropriate purposes.**
> 2. **Continuing to use this tutorial implies agreement with the related regulations described herein. This tutorial fulfills its obligation to provide guidance and is not responsible for any subsequent issues that may arise.**
> 3. **Please resolve dataset authorization issues yourself. Do not use unauthorized datasets for training! Any issues arising from the use of unauthorized datasets are your own responsibility and have no connection to the repository, the repository maintainers, the svc develop team, or the tutorial authors.**

Specific usage regulations are as follows:

- The content of this tutorial represents personal views only and does not represent the views of the so-vits-svc team or the original authors.
- This tutorial assumes the use of the repository maintained by the so-vits-svc team. Please comply with the open-source licenses of any open-source code involved.
- Any videos based on sovits made and posted on video platforms must clearly indicate in the description the source of the input vocals or audio used for the voice converter. For example, if using someone else's video or audio as the input source after vocal separation, a clear link to the original video or music must be provided. If using your own voice or audio synthesized by other vocal synthesis engines, this must also be indicated in the description.
- Ensure the data sources used to create datasets are legal and compliant, and that data providers are aware of what you are creating and the potential consequences. You are solely responsible for any infringement issues arising from the input sources. When using other commercial vocal synthesis software as input sources, ensure you comply with the software's usage terms. Note that many vocal synthesis engine usage terms explicitly prohibit using them as input sources for conversion!
- Cloud training and inference may involve financial costs. If you are a minor, please obtain permission and understanding from your guardian before proceeding. This tutorial is not responsible for any subsequent issues arising from unauthorized use.
- Local training (especially on less powerful hardware) may require prolonged high-load operation of the device. Ensure proper maintenance and cooling measures for your device.
- Due to equipment reasons, this tutorial has only been tested on Windows systems. For Mac and Linux, ensure you have some problem-solving capability.
- Continuing to use this repository implies agreement with the related regulations described in the README. This README fulfills its obligation to provide guidance and is not responsible for any subsequent issues that may arise.

## 0.2 Hardware Requirements

1. Training **must** be conducted using a GPU! For inference, which can be done via **command-line inference** or **WebUI inference**, either a CPU or GPU can be used if speed is not a primary concern.
2. If you plan to train your own model, prepare an **NVIDIA graphics card with at least 6GB of dedicated memory**.
3. Ensure your computer's virtual memory is set to **at least 30GB**, and it is best if it is set on an SSD, otherwise it will be very slow.
4. For cloud training, it is recommended to use [Google Colab](https://colab.google/), you can configure it according to our provided [sovits4_for_colab.ipynb](./sovits4_for_colab.ipynb).

## 0.3 Preparation

1. Prepare at least 30 minutes (the more, the better!) of **clean vocals** as your training set, with **no background noise and no reverb**. It is best to maintain a **consistent timbre** while singing, ensure a **wide vocal range (the vocal range of the training set determines the range of the trained model!)**, and have an **appropriate loudness**. If possible, perform **loudness matching** using audio processing software such as Audition.
2. **Important!** Download the necessary **base model** for training in advance. Refer to [2.2.2 Pre-trained Base Model](#222-pre-trained-baseline-models-highly-recommended).
3. For inference: Prepare **dry vocals** with **background noise <30dB** and preferably **without reverb or harmonies**.

> [!NOTE]
>
> **Note 1**: Both singing and speaking can be used as training sets, but using speech may lead to **issues with high and low notes during inference (commonly known as range issues/muted sound)**, as the vocal range of the training set largely determines the vocal range of the trained model. Therefore, if your final goal is to achieve singing, it is recommended to use singing vocals as your training set.
>
> **Note 2**: When using a male voice model to infer songs sung by female singers, if there is noticeable muting, try lowering the pitch (usually by 12 semitones, or one octave). Similarly, when using a female voice model to infer songs sung by male singers, you can try raising the pitch.
>
> **✨ Latest Recommendation as of 2024.3.8 ✨**: Currently, the [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) project's TTS (Text-to-Speech) compared to so-vits-svc's TTS requires a smaller training set, has faster training speed, and yields better results. Therefore, if you want to use the speech synthesis function, please switch to [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Consequently, it is recommended to use singing vocals as the training set for this project.

# 1. Environment Dependencies

✨ **Required environment for this project**: [NVIDIA-CUDA](https://developer.nvidia.com/cuda-toolkit) | [Python](https://www.python.org/) = 3.8.9 (this version is recommended) | [Pytorch](https://pytorch.org/get-started/locally/) | [FFmpeg](https://ffmpeg.org/)

✨ **You can also try using my script for one-click environment setup and webUI launch: [so-vits-svc-webUI-QuickStart-bat](https://github.com/SUC-DriverOld/so-vits-svc-webUI-QuickStart-bat)**

## 1.1 so-vits-svc4.1 Source Code

You can download or clone the source code using one of the following methods:

1. **Download the source code ZIP file from the Github project page**: Go to the [so-vits-svc official repository](https://github.com/svc-develop-team/so-vits-svc), click the green `Code` button at the top right, and select `Download ZIP` to download the compressed file. If you need the code from another branch, switch to that branch first. After downloading, extract the ZIP file to any directory, which will serve as your working directory.

2. **Clone the source code using git**: Use the following command:

   ```bash
   git clone https://github.com/svc-develop-team/so-vits-svc.git
   ```

## 1.2 Cuda

- Enter `nvidia-smi.exe` in the command prompt to check the version of your graphics driver and the corresponding CUDA version.

- Go to the [NVIDIA-CUDA](https://developer.nvidia.com/cuda-toolkit) official website to download the CUDA version that matches your system.

- For example, to install [Cuda-11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local), select options based on your system and requirements (generally, local Windows users should choose `Windows`, `x86_64`, `System Version`, `exe(local)`).

- After successful installation, enter `nvcc -V` in the command prompt. If the output is similar to the following, the installation was successful:

  ```bash
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
  Cuda compilation tools, release 11.7, V11.7.64
  Build cuda_11.7.r11.7/compiler.31294372_0
  ```

> [!NOTE]
>
> 1. The CUDA version shown in `nvidia-smi.exe` is backward compatible. For example, if it shows 12.4, you can install any CUDA version <=12.4.
> 2. CUDA needs to be compatible with the version of [Pytorch](#14-pytorch) below.
> 3. To uninstall CUDA: Open Control Panel -> Programs -> Uninstall a program, and uninstall all programs with `NVIDIA CUDA` in their name (there are a total of 5).

## 1.3 Python

- Go to the [Python official website](https://www.python.org/) to download Python 3.8.9 and **add it to the system environment variables**. (If using conda to configure Python and 3.8.9 is unavailable, you can directly select 3.8). Detailed installation methods and adding Path are omitted here, as they can be easily found online.

```bash
# Conda configuration method, replace YOUR_ENV_NAME with the name of the virtual environment you want to create.
conda create -n YOUR_ENV_NAME python=3.8 -y
conda activate YOUR_ENV_NAME
# Ensure you are in this virtual environment before executing any commands!
```

- After installation, enter `python` in the command prompt. If the output is similar to the following, the installation was successful:

  ```bash
  Python 3.8.9 (tags/v3.8.9:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
  Type "help", "copyright", "credits" or "license" for more information.
  >>>
  ```

**Regarding the Python version**: After testing, we found that Python 3.8.9 can stably run this project (though higher versions may also work).

## 1.4 Pytorch

> [!IMPORTANT]
>
> ✨ The Pytorch installed here needs to match the CUDA version installed in [1.2 Cuda](#12-cuda). For example, if I installed CUDA 12.1, I should choose Pytorch version 12.1 for installation. Personally, I used pytorch2.0.1+cu117 without encountering significant issues. Please try higher versions at your own discretion.

- We need to **separately install** `torch`, `torchaudio`, `torchvision` libraries. Go directly to the [Pytorch official website](https://pytorch.org/get-started/locally/), choose the desired version, and copy the command displayed in the "Run this Command" section to the console to install. You can download older versions of Pytorch from [here](https://pytorch.org/get-started/previous-versions/).

- After installing `torch`, `torchaudio`, `torchvision`, use the following command in the cmd console to check if torch can successfully call CUDA. If the last line shows `True`, it's successful; if it shows `False`, it's unsuccessful and you need to reinstall the correct version.

```bash
python
# Press Enter to run
import torch
# Press Enter to run
print(torch.cuda.is_available())
# Press Enter to run
```

> [!NOTE]
>
> 1. If you need to specify the version of `torch` manually, simply add the version number afterward. For example, `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117`.
> 2. When installing Pytorch for CUDA 11.7, you may encounter an error `ERROR: Package 'networkx' requires a different Python: 3.8.9 not in '>=3.9'`. In this case, first execute `pip install networkx==3.0`, and then proceed with the Pytorch installation to avoid similar errors.
> 3. Due to version updates, you may not be able to copy the download link for Pytorch 11.7. In this case, you can directly copy the installation command below to install Pytorch 11.7. Alternatively, you can download older versions from [here](https://pytorch.org/get-started/previous-versions/).

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

## 1.5 Installation of Other Dependencies

> [!IMPORTANT]
> ✨ Before starting the installation of other dependencies, **make sure to download and install** [Visual Studio 2022](https://visualstudio.microsoft.com/) or [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/) (the latter has a smaller size). **Select and install the component package: "Desktop development with C++"**, then execute the modification and wait for the installation to complete.

- Right-click in the folder obtained from [1.1](#11-so-vits-svc41-source-code) and select **Open in Terminal**. Use the following command to first update `pip`, `wheel`, and `setuptools`.

```bash
pip install --upgrade pip==23.3.2 wheel setuptools
```

- Execute the following command to install libraries (**if errors occur, please try multiple times until there are no errors, and all dependencies are installed**). Note that there are three `requirements` txt files in the project folder; here, select `requirements_win.txt`.

```bash
pip install -r requirements_win.txt
```

- After ensuring the installation is **correct and error-free**, use the following command to update `fastapi`, `gradio`, and `pydantic` dependencies:

```bash
pip install --upgrade fastapi==0.84.0
pip install --upgrade pydantic==1.10.12
pip install --upgrade gradio==3.41.2
```

## 1.6 FFmpeg

- Go to the [FFmpeg official website](https://ffmpeg.org/) to download FFmpeg. Unzip it to any location and add the path to the environment variables. Navigate to `.\ffmpeg\bin` (detailed installation methods and adding Path are omitted here, as they can be easily found online).

- After installation, enter `ffmpeg -version` in the cmd console. If the output is similar to the following, the installation was successful:

```bash
ffmpeg version git-2020-08-12-bb59bdb Copyright (c) 2000-2020 the FFmpeg developers
built with gcc 10.2.1 (GCC) 20200805
configuration: a bunch of configuration details here
libavutil      56. 58.100 / 56. 58.100
libavcodec     58.100.100 / 58.100.100
...
```

# 2. Configuration and Training

✨ This section is the most crucial part of the entire tutorial document. It references the [official documentation](https://github.com/svc-develop-team/so-vits-svc#readme) and includes some explanations and clarifications for better understanding.

✨ Before diving into the content of the second section, please ensure that your computer's virtual memory is set to **30GB or above**, preferably on a solid-state drive (SSD). You can search online for specific instructions on how to do this.

## 2.1 Issues Regarding Compatibility with the 4.0 Model

- You can ensure support for the 4.0 model by modifying the `config.json` of the 4.0 model. You need to add the `speech_encoder` field under the `model` section in the `config.json`, as shown below:

```bash
  "model":
  {
    # Other contents omitted

    # "ssl_dim", fill in either 256 or 768, which should match the value below "speech_encoder"
    "ssl_dim": 256,
    # Number of speakers
    "n_speakers": 200,
    # or "vec768l12", but please note that the value here should match "ssl_dim" above. That is, 256 corresponds to vec256l9, and 768 corresponds to vec768l12.
    "speech_encoder":"vec256l9"
    # If you're unsure whether your model is vec768l12 or vec256l9, you can confirm by checking the value of the "gin_channels" field.

    # Other contents omitted
  }
```

## 2.2 Pre-downloaded Model Files

### 2.2.1 Mandatory Items

> [!WARNING]
>
> **You must select one of the following encoders to use:**
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
> - "wavlmbase+"

| Encoder                  | Download Link                                                                                                                                                                                                        | Location                                                                    | Description                                                             |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| contentvec (Recommended) | [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)                                                                                                                              | Place in `pretrain` directory                                               | `vec768l12` and `vec256l9` require this encoder                         |
|                          | [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)                                                                                                                     | Rename to checkpoint_best_legacy_500.pt, then place in `pretrain` directory | Same effect as the above `checkpoint_best_legacy_500.pt` but only 199MB |
| hubertsoft               | [hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)                                                                                                           | Place in `pretrain` directory                                               | Used by so-vits-svc3.0                                                  |
| Whisper-ppg              | [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)                                                                       | Place in `pretrain` directory                                               | Compatible with `whisper-ppg`                                           |
| whisper-ppg-large        | [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)                                                                   | Place in `pretrain` directory                                               | Compatible with `whisper-ppg-large`                                     |
| cnhubertlarge            | [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)                                                                | Place in `pretrain` directory                                               | -                                                                       |
| dphubert                 | [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)                                                                                                                        | Place in `pretrain` directory                                               | -                                                                       |
| WavLM                    | [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) | Place in `pretrain` directory                                               | Download link might be problematic, unable to download                  |
| OnnxHubert/ContentVec    | [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)                                                                                                                                 | Place in `pretrain` directory                                               | -                                                                       |

#### Detailed Explanation of Each Encoder

| Encoder Name                   | Advantages                                                         | Disadvantages                     |
| ------------------------------ | ------------------------------------------------------------------ | --------------------------------- |
| `vec768l12` (Most Recommended) | Best voice fidelity, large base model, supports loudness embedding | Weak articulation                 |
| `vec256l9`                     | No particular advantages                                           | Does not support diffusion models |
| `hubertsoft`                   | Strong articulation                                                | Voice leakage                     |
| `whisper-ppg`                  | Strongest articulation                                             | Voice leakage, high VRAM usage    |

### 2.2.2 Pre-trained Base Model (Strongly Recommended)

- Pre-trained base model files: `G_0.pth`, `D_0.pth`. Place in the `logs/44k` directory.

- Diffusion model pre-trained base model file: `model_0.pt`. Place in the `logs/44k/diffusion` directory.

The diffusion model references the Diffusion Model from [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC), and the base model is compatible with the diffusion model from [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC). Some of the provided base model files are from the integration package of “[羽毛布団](https://space.bilibili.com/3493141443250876)”, to whom we express our gratitude.

**Provide 4.1 training base models, please download them yourself (requires external network conditions)**

| Encoder Type                        | Main Model Base                                                                                                                                                                                                                  | Diffusion Model Base                                                                                                 | Description                                                                                                                                                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| vec768l12                           | [G_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/G_0.pth), [D_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/D_0.pth)                           | [model_0.pt](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/768l12/model_0.pt)      | If only training for 100 steps diffusion, i.e., `k_step_max = 100`, use [model_0.pt](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/768l12/max100/model_0.pt) for the diffusion model |
| vec768l12 (with loudness embedding) | [G_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/vol_emb/G_0.pth), [D_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec768l12/vol_emb/D_0.pth)           | [model_0.pt](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/768l12/model_0.pt)      | If only training for 100 steps diffusion, i.e., `k_step_max = 100`, use [model_0.pt](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/768l12/max100/model_0.pt) for the diffusion model |
| vec256l9                            | [G_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec256l9/G_0.pth), [D_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/vec256l9/D_0.pth)                             | Not supported                                                                                                        | -                                                                                                                                                                                                                      |
| hubertsoft                          | [G_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/hubertsoft/G_0.pth), [D_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/hubertsoft/D_0.pth)                         | [model_0.pt](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/hubertsoft/model_0.pt)  | -                                                                                                                                                                                                                      |
| whisper-ppg                         | [G_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/whisper-ppg/G_0.pth), [D_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/whisper-ppg/D_0.pth)                       | [model_0.pt](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/diffusion/whisper-ppg/model_0.pt) | -                                                                                                                                                                                                                      |
| tiny (vec768l12_vol_emb)            | [G_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/tiny/vec768l12_vol_emb/G_0.pth), [D_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/blob/main/tiny/vec768l12_vol_emb/D_0.pth) | -                                                                                                                    | TINY is based on the original So-VITS model with reduced network parameters, using Depthwise Separable Convolution and FLOW                                                                                            |

shared parameter technology, significantly reducing model size and improving inference speed. TINY is designed for real-time conversion; reduced parameters mean its conversion effect is theoretically inferior to the original model. Real-time conversion GUI for So-VITS is under development. Until then, if there's no special need, training TINY model is not recommended. |

> [!WARNING]
>
> Pre-trained models for other encoders not mentioned are not provided. Please train without base models, which may significantly increase training difficulty!

**Base Model and Support**

| Standard Base | Loudness Embedding | Loudness Embedding + TINY | Full Diffusion | 100-Step Shallow Diffusion |
| ------------- | ------------------ | ------------------------- | -------------- | -------------------------- |
| Vec768L12     | Supported          | Supported                 | Supported      | Supported                  |
| Vec256L9      | Supported          | Not Supported             | Not Supported  | Not Supported              |
| hubertsoft    | Supported          | Not Supported             | Supported      | Not Supported              |
| whisper-ppg   | Supported          | Not Supported             | Supported      | Not Supported              |

### 2.2.3 Optional Items (Choose as Needed)

**1. NSF-HIFIGAN**

If using the `NSF-HIFIGAN enhancer` or `shallow diffusion`, you need to download the pre-trained NSF-HIFIGAN model provided by [OpenVPI]. If not needed, you can skip this.

- Pre-trained NSF-HIFIGAN vocoder:
  - Version 2022.12: [nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip);
  - Version 2024.02: [nsf_hifigan_44.1k_hop512_128bin_2024.02.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-44.1k-hop512-128bin-2024.02/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip)
- After extracting, place the four files in the `pretrain/nsf_hifigan` directory.
- If you download version 2024.02 of the vocoder, rename `model.ckpt` to `model`, i.e., remove the file extension.

**2. RMVPE**

If using the `rmvpe` F0 predictor, you need to download the pre-trained RMVPE model.

- Download the model [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip), which is currently recommended.
- Extract `rmvpe.zip`, rename the `model.pt` file to `rmvpe.pt`, and place it in the `pretrain` directory.

**3. FCPE (Preview Version)**

[FCPE](https://github.com/CNChTu/MelPE) (Fast Context-based Pitch Estimator) is a new F0 predictor developed independently by svc-develop-team, designed specifically for real-time voice conversion. It will become the preferred F0 predictor for Sovits real-time voice conversion in the future.

If using the `fcpe` F0 predictor, you need to download the pre-trained FCPE model.

- Download the model [fcpe.pt](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt).
- Place it in the `pretrain` directory.

## 2.3 Data Preparation

1. Organize the dataset into the `dataset_raw` directory according to the following file structure.

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

2. You can customize the names of the speakers.

```
dataset_raw
└───suijiSUI
    ├───1.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

3. Additionally, you need to create and edit `config.json` in `dataset_raw`

```json
{
  "n_speakers": 10,

  "spk": {
    "speaker0": 0,
    "speaker1": 1
  }
}
```

- `"n_speakers": 10`: The number represents the number of speakers, starting from 1, and needs to correspond to the number below
- `"speaker0": 0`: "speaker0" refers to the speaker's name, which can be changed. The numbers 0, 1, 2... represent the speaker count, starting from 0.

## 2.4 Data Preprocessing

### 2.4.0 Audio Slicing

- Slice the audio into `5s - 15s` segments. Slightly longer segments are acceptable, but excessively long segments may lead to out-of-memory errors during training or preprocessing.

- You can use [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer) or [audio-slicer-CLI](https://github.com/openvpi/audio-slicer) for assistance in slicing. Generally, you only need to adjust the `Minimum Interval`. For regular speech material, the default value is usually sufficient, while for singing material, you may adjust it to `100` or even `50`.

- After slicing, manually handle audio that is too long (over 15 seconds) or too short (under 4 seconds). Short audio can be concatenated into multiple segments, while long audio can be manually split.

> [!WARNING]
>
> **If you are training with the Whisper-ppg sound encoder, all slices must be less than 30s in length.**

### 2.4.1 Resampling to 44100Hz Mono

Use the following command (skip this step if loudness matching has already been performed):

```bash
python resample.py
```

> [!NOTE]
>
> Although this project provides a script `resample.py` for resampling, converting to mono, and loudness matching, the default loudness matching matches to 0db, which may degrade audio quality. Additionally, the loudness normalization package `pyloudnorm` in Python cannot apply level limiting, which may lead to clipping. It is recommended to consider using professional audio processing software such as `Adobe Audition` for loudness matching. You can also use a loudness matching tool I developed, [Loudness Matching Tool](https://github.com/AI-Hobbyist/Loudness-Matching-Tool). If you have already performed loudness matching with other software, you can add `--skip_loudnorm` when running the above command to skip the loudness matching step. For example:

```bash
python resample.py --skip_loudnorm
```

### 2.4.2 Automatic Dataset Splitting and Configuration File Generation

Use the following command (skip this step if loudness embedding is required):

```bash
python preprocess_flist_config.py --speech_encoder vec768l12
```

The `speech_encoder` parameter has seven options, as explained in **[2.2.1 Required Items and Explanation of Each Encoder](#detailed-explanation-of-each-encoder)**. If you omit the `speech_encoder` parameter, the default value is `vec768l12`.

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
whisper-ppg-large
cnhubertlarge
dphubert
```

#### Using Loudness Embedding

- When using loudness embedding, the trained model will match the loudness of the input source. Otherwise, it will match the loudness of the training set.
- If using loudness embedding, you need to add the `--vol_aug` parameter, for example:

```bash
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

### 2.4.3 Modify Configuration Files as Needed

#### config.json

- `vocoder_name`: Select a vocoder, default is `nsf-hifigan`.
- `log_interval`: How often to output logs, default is `200`.
- `eval_interval`: How often to perform validation and save the model, default is `800`.
- `epochs`: Total number of training epochs, default is `10000`. Training will automatically stop after reaching this number of epochs.
- `learning_rate`: Learning rate, it's recommended to keep the default value.
- `batch_size`: The amount of data loaded onto the GPU for each training step, adjust to a size lower than the GPU memory capacity.
- `all_in_mem`: Load all dataset into memory. Enable this if disk IO is too slow on some platforms and the memory capacity is much larger than the dataset size.
- `keep_ckpts`: Number of recent models to keep during training, `0` to keep all. Default is to keep only the last `3` models.

**Vocoder Options**

```
nsf-hifigan
nsf-snake-hifigan
```

#### diffusion.yaml

- `cache_all_data`: Load all dataset into memory. Enable this if disk IO is too slow on some platforms and the memory capacity is much larger than the dataset size.
- `duration`: Duration of audio slices during training. Adjust according to GPU memory size. **Note: This value must be less than the shortest duration of audio in the training set!**
- `batch_size`: The amount of data loaded onto the GPU for each training step, adjust to a size lower than the GPU memory capacity.
- `timesteps`: Total steps of the diffusion model, default is 1000. A complete Gaussian diffusion has a total of 1000 steps.
- `k_step_max`: During training, only `k_step_max` steps of diffusion can be trained to save training time. Note that this value must be less than `timesteps`. `0` means training the entire diffusion model. **Note: If not training the entire diffusion model, the model can't be used for inference with only the diffusion model!**

### 2.4.3 Generating Hubert and F0

Use the following command (skip this step if training shallow diffusion):

```bash
# The following command uses rmvpe as the f0 predictor, you can manually modify it
python preprocess_hubert_f0.py --f0_predictor rmvpe
```

The `f0_predictor` parameter has six options, and some F0 predictors require downloading additional preprocessing models. Please refer to **[2.2.3 Optional Items (Choose According to Your Needs)](#223-optional-items-choose-as-needed)** for details.

```
crepe
dio
pm
harvest
rmvpe (recommended!)
fcpe
```

#### Pros and Cons of Each F0 Predictor

| Predictor | Pros                                                                                          | Cons                                                                     |
| --------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| pm        | Fast, low resource consumption                                                                | Prone to producing breathy voice                                         |
| crepe     | Rarely produces breathy voice                                                                 | High memory consumption, may produce out-of-tune voice                   |
| dio       | -                                                                                             | May produce out-of-tune voice                                            |
| harvest   | Better performance in the lower pitch range                                                   | Inferior performance in other pitch ranges                               |
| rmvpe     | Almost flawless, currently the most accurate predictor                                        | Virtually no drawbacks (extremely low-pitched sounds may be problematic) |
| fcpe      | Developed by the SVC team, currently the fastest predictor, with accuracy comparable to crepe | -                                                                        |

> [!NOTE]
>
> 1. If the training set is too noisy, use crepe to process f0.
> 2. If you omit the `f0_predictor` parameter, the default value is rmvpe.

**If shallow diffusion functionality is required (optional), add the --use_diff parameter, for example:**

```bash
# The following command uses rmvpe as the f0 predictor, you can manually modify it
python preprocess_hubert_f0.py --f0_predictor rmvpe --use_diff
```

**If the processing speed is slow, or if your dataset is large, you can add the `--num_processes` parameter:**

```bash
# The following command uses rmvpe as the f0 predictor, you can manually change it
python preprocess_hubert_f0.py --f0_predictor rmvpe --num_processes 8
# All workers will be automatically assigned to multiple threads
```

After completing the above steps, the `dataset` directory generated is the preprocessed data, and you can delete the `dataset_raw` folder as needed.

## 2.5 Training

### 2.5.1 Main Model Training (Required)

Use the following command to train the main model. You can also use the same command to resume training if it pauses.

```bash
python train.py -c configs/config.json -m 44k
```

### 2.5.2 Diffusion Model (Optional)

- A major update in So-VITS-SVC 4.1 is the introduction of Shallow Diffusion mechanism, which converts the original output audio of SoVITS into Mel spectrograms, adds noise, and performs shallow diffusion processing before outputting the audio through the vocoder. Testing has shown that **the quality of the output is significantly enhanced after the original output audio undergoes shallow diffusion processing, addressing issues such as electronic noise and background noise**.
- If shallow diffusion functionality is required, you need to train the diffusion model. Before training, ensure that you have downloaded and correctly placed `NSF-HIFIGAN` (**refer to [2.2.3 Optional Items](#223-optional-items-choose-as-needed)**), and added the `--use_diff` parameter when preprocessing to generate Hubert and F0 (**refer to [2.4.3 Generating Hubert and F0](#243-generating-hubert-and-f0)**).

To train the diffusion model, use the following command:

```bash
python train_diff.py -c configs/diffusion.yaml
```

After the model training is complete, the model files are saved in the `logs/44k` directory, and the diffusion model is saved in `logs/44k/diffusion`.

> [!IMPORTANT]
>
> **How do you know when the model is trained well?**
>
> 1. This is a very boring and meaningless question. It's like asking a teacher how to make your child study well. Except for yourself, no one can answer this question.
> 2. The model training is related to the quality and duration of your dataset, the selected encoder, f0 algorithm, and even some supernatural mystical factors. Even if you have a finished model, the final conversion effect depends on your input source and inference parameters. This is not a linear process, and there are too many variables involved. So, if you have to ask questions like "Why doesn't my model look like it?" or "How do you know when the model is trained well?", I can only say WHO F\*\*KING KNOWS?
> 3. But that doesn't mean there's no way. You just have to pray and worship. I don't deny that praying and worshiping is an effective method, but you can also use some scientific tools, such as Tensorboard, etc. [2.5.3 Tensorboard](#253-tensorboard) below will teach you how to use Tensorboard to assist in understanding the training status. **Of course, the most powerful tool is actually within yourself. How do you know when a acoustic model is trained well? Put on your headphones and let your ears tell you.**

**Relationship between Epoch and Step**:

During training, a model will be saved every specified number of steps (default is 800 steps, corresponding to the `eval_interval` value) based on the setting in your `config.json`.
It's important to distinguish between epochs and steps: 1 Epoch means all samples in the training set have been involved in one learning process, while 1 Step means one learning step has been taken. Due to the existence of `batch_size`, each learning step can contain several samples. Therefore, the conversion between Epoch and Step is as follows:

$$
Epoch = \frac{Step}{(\text{Number of samples in the dataset}{\div}batch\_size)}
$$

The training will end after 10,000 epochs by default (you can increase or decrease the upper limit by modifying the value of the `epoch` field in `config.json`), but typically, good results can be achieved after a few hundred epochs. When you feel that training is almost complete, you can interrupt the training by pressing `Ctrl + C` in the training terminal. After interruption, as long as you haven't reprocessed the training set, you can **resume training from the most recent saved point**.

### 2.5.3 Tensorboard

You can use Tensorboard to visualize the trends of loss function values during training, listen to audio samples, and assist in judging the training status of the model. **However, for the So-VITS-SVC project, the loss function values (loss) do not have practical reference significance (you don't need to compare or study the value itself), the real reference is still listening to the audio output after inference with your ears!**

- Use the following command to open Tensorboard:

```bash
tensorboard --logdir=./logs/44k
```

Tensorboard generates logs based on the default evaluation every 200 steps during training. If training has not reached 200 steps, no images will appear in Tensorboard. The value of 200 can be modified by changing the value of `log_interval` in `config.json`.

- Explanation of Losses

You don't need to understand the specific meanings of each loss. In general:

- `loss/g/f0`, `loss/g/mel`, and `loss/g/total` should oscillate and eventually converge to some value.
- `loss/g/kl` should oscillate at a low level.
- `loss/g/fm` should continue to rise in the middle of training, and in the later stages, the upward trend should slow down or even start to decline.

> [!IMPORTANT]
>
> ✨ Observing the trends of loss curves can help you judge the training status of the model. However, losses alone cannot be the sole criterion for judging the training status of the model, **and in fact, their reference value is not very significant. You still need to judge whether the model is trained well by listening to the audio output with your ears**.

> [!WARNING]
>
> 1. For small datasets (30 minutes or even smaller), it is not recommended to train for too long when loading the base model. This is to make the best use of the advantages of the base model. The best results can be achieved in thousands or even hundreds of steps.
> 2. The audio samples in Tensorboard are generated based on your validation set and **cannot represent the final performance of the model**.

# 3. Inference

✨ Before inference, please prepare the dry audio you need for inference, ensuring it has no background noise/reverb and is of good quality. You can use [UVR5](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.6) for processing to obtain the dry audio. Additionally, I've also created a [UVR5 vocal separation tutorial](https://www.bilibili.com/video/BV1F4421c7qU/).

## 3.1 Command-line Inference

Perform inference using inference_main.py

```bash
# Example
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "your_inference_audio.wav" -t 0 -s "speaker"
```

**Required Parameters:**

- `-m` | `--model_path`: Path to the model
- `-c` | `--config_path`: Path to the configuration file
- `-n` | `--clean_names`: List of wav file names, placed in the raw folder
- `-t` | `--trans`: Pitch adjustment, supports positive and negative (in semitones)
- `-s` | `--spk_list`: Name of the target speaker for synthesis
- `-cl` | `--clip`: Audio forced clipping, default 0 for automatic clipping, unit in seconds/s.

> [!NOTE]
>
> **Audio Clipping**
>
> - During inference, the clipping tool will split the uploaded audio into several small segments based on silence sections, and then combine them after inference to form the complete audio. This approach benefits from lower GPU memory usage for small audio segments, thus enabling the segmentation of long audio for inference to avoid GPU memory overflow. The clipping threshold parameter controls the minimum full-scale decibel value, and anything lower will be considered as silence and removed. Therefore, when the uploaded audio is noisy, you can set this parameter higher (e.g., -30), whereas for cleaner audio, a lower value (e.g., -50) can be set to avoid cutting off breath sounds and faint voices.
>
> - A recent test by the development team suggests that smaller clipping thresholds (e.g., -50) improve the clarity of the output, although the principle behind this is currently unclear.
>
> **Forced Clipping** `-cl` | `--clip`
>
> - During inference, the clipping tool may sometimes produce overly long audio segments when continuous vocal sections exist without silence for an extended period, potentially causing GPU memory overflow. The automatic audio clipping feature sets a maximum duration for audio segmentation. After the initial segmentation, if there are audio segments longer than this duration, they will be forcibly re-segmented at this duration to avoid memory overflow issues.
> - Forced clipping may result in cutting off audio in the middle of a word, leading to discontinuity in the synthesized voice. You need to set the crossfade length for forced clipping in advanced settings to mitigate this issue.

**Optional Parameters: See Next Section for Specifics**

- `-lg` | `--linear_gradient`: Crossfade length of two audio clips, adjust this value if there are discontinuities in the voice after forced clipping, recommended to use default value 0, unit in seconds
- `-f0p` | `--f0_predictor`: Choose F0 predictor, options are crepe, pm, dio, harvest, rmvpe, fcpe, default is pm (Note: crepe uses mean filter for original F0), refer to the advantages and disadvantages of different F0 predictors in [2.4.3 F0 Predictor Advantages and Disadvantages](#243-generating-hubert-and-f0)
- `-a` | `--auto_predict_f0`: Automatically predict pitch during voice conversion, do not enable this when converting singing voices as it may severely mis-tune
- `-cm` | `--cluster_model_path`: Path to clustering model or feature retrieval index, leave empty to automatically set to the default path of each solution model, fill in randomly if no clustering or feature retrieval is trained
- `-cr` | `--cluster_infer_ratio`: Ratio of clustering solution or feature retrieval, range 0-1, defaults to 0 if no clustering model or feature retrieval is trained
- `-eh` | `--enhance`: Whether to use the NSF_HIFIGAN enhancer, this option has a certain sound quality enhancement effect on models with a limited training set, but has a negative effect on well-trained models, default is off
- `-shd` | `--shallow_diffusion`: Whether to use shallow diffusion, enabling this can solve some electronic sound problems, default is off, when this option is enabled, the NSF_HIFIGAN enhancer will be disabled
- `-usm` | `--use_spk_mix`: Whether to use speaker blending/dynamic voice blending
- `-lea` | `--loudness_envelope_adjustment`: Ratio of input source loudness envelope replacement to output loudness envelope fusion, the closer to 1, the more the output loudness envelope is used
- `-fr` | `--feature_retrieval`: Whether to use feature retrieval, if a clustering model is used, it will be disabled, and the cm and cr parameters will become the index path and mixing ratio of feature retrieval

> [!NOTE]
>
> **Clustering Model/Feature Retrieval Mixing Ratio** `-cr` | `--cluster_infer_ratio`
>
> - This parameter controls the proportion of linear involvement when using clustering models/feature retrieval models. Clustering models and feature retrieval models can both slightly improve timbre similarity, but at the cost of reducing accuracy in pronunciation (feature retrieval has slightly better pronunciation than clustering). The range of this parameter is 0-1, where 0 means it is not enabled, and the closer to 1, the more similar the timbre and the blurrier the pronunciation.
> - Clustering models and feature retrieval share this parameter. When loading models, the model used will be controlled by this parameter.
> - **Note that when clustering models or feature retrieval models are not loaded, please keep this parameter as 0, otherwise an error will occur.**

**Shallow Diffusion Settings:**

- `-dm` | `--diffusion_model_path`: Diffusion model path
- `-dc` | `--diffusion_config_path`: Diffusion model configuration file path
- `-ks` | `--k_step`: Number of diffusion steps, larger values are closer to the result of the diffusion model, default is 100
- `-od` | `--only_diffusion`: Pure diffusion mode, this mode does not load the sovits model and performs inference based only on the diffusion model
- `-se` | `--second_encoding`: Secondary encoding, the original audio will be encoded a second time before shallow diffusion, a mysterious option, sometimes it works well, sometimes it doesn't

> [!NOTE]
>
> **About Shallow Diffusion Steps** `-ks` | `--k_step`
>
> The complete Gaussian diffusion takes 1000 steps. When the number of shallow diffusion steps reaches 1000, the output result at this point is entirely the output result of the diffusion model, and the So-VITS model will be suppressed. The higher the number of shallow diffusion steps, the closer it is to the output result of the diffusion model. **If you only want to use shallow diffusion to remove electronic noise while preserving the timbre of the So-VITS model as much as possible, the number of shallow diffusion steps can be set to 30-50**

> [!WARNING]
>
> If using the `whisper-ppg` voice encoder for inference, `--clip` should be set to 25, `-lg` should be set to 1. Otherwise, inference will not work properly.

## 3.2 webUI Inference

Use the following command to open the webUI interface, **upload and load the model, fill in the inference as needed according to the instructions, upload the inference audio, and start the inference.**

The detailed explanation of the inference parameters is the same as the [3.1 Command-line Inference](#31-command-line-inference) parameters, but moved to the interactive interface with simple instructions.

```bash
python webUI.py
```

> [!WARNING]
>
> **Be sure to check [Command-line Inference](#31-command-line-inference) to understand the meanings of specific parameters. Pay special attention to the reminders in NOTE and WARNING!**

The webUI also has a built-in **text-to-speech (TTS)** function:

- Text-to-speech uses Microsoft's edge_TTS service to generate a piece of original speech, and then converts the voice of this speech to the target voice using So-VITS. So-VITS can only achieve voice conversion (SVC) for singing voices, and does not have any **native** text-to-speech (TTS) function! Since the speech generated by Microsoft's edge_TTS is relatively stiff and lacks emotion, all converted audio will also reflect this. **If you need a TTS function with emotions, please visit the [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) project.**
- Currently, text-to-speech supports a total of 55 languages, covering most common languages. The program will automatically recognize the language based on the text entered in the text box and convert it.
- Automatic recognition can only recognize the language, and certain languages may have different accents, speakers. If automatic recognition is used, the program will randomly select one from the speakers that fit the language and specified gender for conversion. If your target language has multiple accents or speakers (e.g., English), it is recommended to manually specify one speaker with a specific accent. If a speaker is manually specified, the previously manually selected gender will be suppressed.

# 4. Optional Enhancements

✨ If you are satisfied with the previous effects, or didn't quite understand what's being discussed below, you can ignore the following content without affecting model usage (these optional enhancements have relatively minor effects, and may only have some effect on specific data, but in most cases, the effect is not very noticeable).

## 4.1 Automatic F0 Prediction

During model training, an F0 predictor is trained, which is an automatic pitch shifting function that can match the model pitch to the source pitch during inference, useful for voice conversion where it can better match the pitch. **However, do not enable this feature when converting singing voices!!! It will severely mis-tune!!!**

- Command-line Inference: Set `auto_predict_f0` to `true` in `inference_main`.
- WebUI Inference: Check the corresponding option.

## 4.2 Clustering Timbre Leakage Control

Clustering schemes can reduce timbre leakage, making the model output more similar to the target timbre (though not very obvious). However, using clustering alone can reduce the model's pronunciation clarity (making it unclear), this model adopts a fusion approach, allowing linear control of the proportion of clustering schemes and non-clustering schemes. In other words, you can manually adjust the ratio between "similar to target timbre" and "clear pronunciation" to find a suitable balance.

Using clustering does not require any changes to the existing steps mentioned earlier, just need to train an additional clustering model, although the effect is relatively limited, the training cost is also relatively low.

- Training Method:

```bash
# Train using CPU:
python cluster/train_cluster.py
# Or train using GPU:
python cluster/train_cluster.py --gpu
```

After training, the model output will be saved in `logs/44k/kmeans_10000.pt`

- During Command-line Inference:
  - Specify `cluster_model_path` in `inference_main.py`
  - Specify `cluster_infer_ratio` in `inference_main.py`, where `0` means not using clustering at all, `1` means only using clustering. Usually, setting `0.5` is sufficient.
- During WebUI Inference:
  - Upload and load the clustering model.
  - Set the clustering model/feature retrieval mixing ratio, between 0-1, where 0 means not using clustering/feature retrieval at all. Using clustering/feature retrieval can improve timbre similarity but may result in reduced pronunciation clarity (if used, it's recommended to set around 0.5).

## 4.3 Feature Retrieval

Similar to clustering schemes, feature retrieval can also reduce timbre leakage, with slightly better pronunciation clarity than clustering, but it may reduce inference speed. Adopting a fusion approach, it allows linear control of the proportion of feature retrieval and non-feature retrieval.

- Training Process: After generating hubert and f0, execute:

```bash
python train_index.py -c configs/config.json
```

After training, the model output will be saved in `logs/44k/feature_and_index.pkl`

- During Command-line Inference:
  - Specify `--feature_retrieval` first, and the clustering scheme will automatically switch to the feature retrieval scheme
  - Specify `cluster_model_path` in `inference_main.py` as the model output file
  - Specify `cluster_infer_ratio` in `inference_main.py`, where `0` means not using feature retrieval at all, `1` means only using feature retrieval. Usually, setting `0.5` is sufficient.
- During WebUI Inference:
  - Upload and load the clustering model
  - Set the clustering model/feature retrieval mixing ratio, between 0-1, where 0 means not using clustering/feature retrieval at all. Using clustering/feature retrieval can improve timbre similarity but may result in reduced pronunciation clarity (if used, it's recommended to set around 0.5)

## 4.4 Vocoder Fine-tuning

When using the diffusion model in So-VITS, the Mel spectrogram enhanced by the diffusion model is output as the final audio through the vocoder. The vocoder plays a decisive role in the sound quality of the output audio. So-VITS-SVC currently uses the [NSF-HiFiGAN community vocoder](https://openvpi.github.io/vocoders/). In fact, you can also fine-tune this vocoder model with your own dataset to better suit your model task in the **diffusion process** of So-VITS.

The [SingingVocoders](https://github.com/openvpi/SingingVocoders) project provides methods for fine-tuning the vocoder. In the Diffusion-SVC project, **using a fine-tuned vocoder can greatly enhance the output sound quality**. You can also train a fine-tuned vocoder with your own dataset and use it in this integration package.

1. Train a fine-tuned vocoder using [SingingVocoders](https://github.com/openvpi/SingingVocoders) and obtain its model and configuration files.
2. Place the model and configuration files under `pretrain/{fine-tuned vocoder name}/`.
3. Modify the diffusion model configuration file `diffusion.yaml` of the corresponding model as follows:

```yaml
vocoder:
  ckpt: pretrain/nsf_hifigan/model.ckpt # This line is the path to your fine-tuned vocoder model
  type: nsf-hifigan # This line is the type of your fine-tuned vocoder, do not modify if unsure
```

1. Following [3.2 webUI Inference](#32-webui-inference), upload the diffusion model and the **modified diffusion model configuration file** to use the fine-tuned vocoder.

> [!WARNING]
>
> **Currently, only the NSF-HiFiGAN vocoder supports fine-tuning.**

## 4.5 Directories for Saved Models

Up to the previous section, a total of 4 types of models that can be trained have been covered. The following table summarizes these four types of models and their configuration files.

In the webUI, in addition to uploading and loading models, you can also read local model files. You just need to put these models into a folder first, and then put the folder into the `trained` folder. Click "Refresh Local Model List", and the webUI will recognize it. Then manually select the model you want to load for loading. **Note**: Automatic loading of local models may not work properly for the (optional) models in the table below.

| File                                          | Filename and Extension  | Location             |
| --------------------------------------------- | ----------------------- | -------------------- |
| So-VITS Model                                 | `G_xxxx.pth`            | `logs/44k`           |
| So-VITS Model Configuration File              | `config.json`           | `configs`            |
| Diffusion Model (Optional)                    | `model_xxxx.pt`         | `logs/44k/diffusion` |
| Diffusion Model Configuration File (Optional) | `diffusion.yaml`        | `configs`            |
| Kmeans Clustering Model (Optional)            | `kmeans_10000.pt`       | `logs/44k`           |
| Feature Retrieval Model (Optional)            | `feature_and_index.pkl` | `logs/44k`           |

# 5. Other Optional Features

✨ This part is less important compared to the previous sections. Except for [5.1 Model Compression](#51-model-compression), which is a more convenient feature, the probability of using the other optional features is relatively low. Therefore, only references to the official documentation and brief descriptions are provided here.

## 5.1 Model Compression

The generated models contain information needed for further training. If you are **sure not to continue training**, you can remove this part of the information from the model to obtain a final model that is about 1/3 of the size.

Use `compress_model.py`

```bash
# For example, if I want to compress a model named G_30400.pth under the logs/44k/ directory, and the configuration file is configs/config.json, I can run the following command
python compress_model.py -c="configs/config.json" -i="logs/44k/G_30400.pth" -o="logs/44k/release.pth"
# The compressed model is saved in logs/44k/release.pth
```

> [!WARNING]
>
> **Note: Compressed models cannot be further trained!**

## 5.2 Voice Mixing

### 5.2.1 Static Voice Mixing

**Refer to the static voice mixing feature in the `webUI.py` file under the Tools/Experimental Features.**

This feature can combine multiple voice models into one voice model (convex combination or linear combination of multiple model parameters), thus creating voice characteristics that do not exist in reality.

**Note:**

1. This feature only supports single-speaker models.
2. If you forcibly use multi-speaker models, you need to ensure that the number of speakers in multiple models is the same, so that voices under the same SpaekerID can be mixed.
3. Ensure that the `model` field in the config.json of all models to be mixed is the same.
4. The output mixed model can use any config.json of the models to be mixed, but the clustering model will not be available.
5. When uploading models in batches, it is better to put the models in a folder and upload them together.
6. It is recommended to adjust the mixing ratio between 0 and 100. Other numbers can also be adjusted, but unknown effects may occur in linear combination mode.
7. After mixing, the file will be saved in the project root directory with the filename output.pth.
8. Convex combination mode will execute Softmax on the mixing ratio to ensure that the sum of mixing ratios is 1, while linear combination mode will not.

### 5.2.2 Dynamic Voice Mixing

**Refer to the introduction of dynamic voice mixing in the `spkmix.py` file.**

Rules for mixing role tracks:

- Speaker ID: \[\[Start Time 1, End Time 1, Start Value 1, End Value 1], [Start Time 2, End Time 2, Start Value 2, End Value 2]]
- The start time must be the same as the end time of the previous one, and the first start time must be 0, and the last end time must be 1 (the time range is 0-1).
- All roles must be filled in, and roles that are not used can be filled with \[\[0., 1., 0., 0.]].
- The fusion value can be filled arbitrarily. Within the specified time range, it changes linearly from the start value to the end value. The internal will automatically ensure that the linear combination is 1 (convex combination condition), so you can use it with confidence.

Use the `--use_spk_mix` parameter during command line inference to enable dynamic voice mixing. Check the "Dynamic Voice Mixing" option box during webUI inference.

## 5.3 Onnx Export

Use `onnx_export.py`. Currently, only [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio) requires the use of onnx models. For more detailed operations and usage methods, please refer to the [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio) repository instructions.

- Create a new folder: `checkpoints` and open it
- In the `checkpoints` folder, create a folder as the project folder, named after your project, such as `aziplayer`
- Rename your model to `model.pth` and the configuration file to `config.json`, and place them in the `aziplayer` folder you just created
- Change `"NyaruTaffy"` in `path = "NyaruTaffy"` in `onnx_export.py` to your project name, `path = "aziplayer" (onnx_export_speaker_mix, for onnx export supporting role mixing)`
- Run `python onnx_export.py`
- Wait for execution to complete. A `model.onnx` will be generated in your project folder, which is the exported model.

Note: Use onnx models provided by [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio) for Hubert Onnx models. Currently, it cannot be exported independently (Hubert in fairseq has many operators not supported by onnx and involves constants, which will cause errors or problems with the input and output shapes and results during export).

# 6. Simple Mixing and Exporting Finished Product

### Use Audio Host Software to Process Inferred Audio

Please refer to the [corresponding video tutorial](https://www.bilibili.com/video/BV1Hr4y197Cy/) or other professional mixing tutorials for details on how to handle and enhance the inferred audio using audio host software.

# Appendix: Common Errors and Solutions

✨ **Some error solutions are credited to [羽毛布団](https://space.bilibili.com/3493141443250876)'s [related column](https://www.bilibili.com/read/cv22206231) | [related documentation](https://www.yuque.com/umoubuton/ueupp5/ieinf8qmpzswpsvr)**

## About Out of Memory (OOM)

If you encounter an error like this in the terminal or WebUI:

```bash
OutOfMemoryError: CUDA out of memory. Tried to allocate XX GiB (GPU 0: XX GiB total capacity; XX GiB already allocated; XX MiB Free; XX GiB reserved in total by PyTorch)
```

Don't doubt it, your GPU memory or virtual memory is insufficient. The following steps provide a 100% solution to the problem. Follow these steps to resolve the issue. Please avoid asking this question in various places as the solution is well-documented.

1. In the error message, find if `XX GiB already allocated` is followed by `0 bytes free`. If it shows `0 bytes free`, follow steps 2, 3, and 4. If it shows `XX MiB free` or `XX GiB free`, follow step 5.
2. If the out of memory occurs during preprocessing:
   - Use a GPU-friendly F0 predictor (from highest to lowest friendliness: pm >= harvest >= rmvpe ≈ fcpe >> crepe). It is recommended to use rmvpe or fcpe first.
   - Set multi-process preprocessing to 1.
3. If the out of memory occurs during training:
   - Check if there are any excessively long clips in the dataset (more than 20 seconds).
   - Reduce the batch size.
   - Use a project with lower resource requirements.
   - Rent a GPU with larger memory from platforms like Google Colab for training.
4. If the out of memory occurs during inference:
   - Ensure the source audio (dry vocal) is clean (no residual reverb, accompaniment, or harmony) as dirty sources can hinder automatic slicing. Refer to the [UVR5 vocal separation tutorial](https://www.bilibili.com/video/BV1F4421c7qU/) for best practices.
   - Increase the slicing threshold (e.g., change from -40 to -30; avoid going too high as it can cut the audio abruptly).
   - Set forced slicing, starting from 60 seconds and decreasing by 10 seconds each time until inference succeeds.
   - Use CPU for inference, which will be slower but won't encounter out of memory issues.
5. If there is still available memory but the out of memory error persists, increase your virtual memory to at least 50G.

These steps should help you manage and resolve out of memory errors effectively, ensuring smooth operation during preprocessing, training, and inference.

## Common Errors and Solutions When Installing Dependencies

**1. Error When Installing PyTorch with CUDA=11.7**

```
ERROR: Package 'networkx' requires a different Python: 3.8.9 not in '>=3.9
```

**Solutions:**

- **Upgrade Python to 3.9:** This might cause instability in some cases.
- **Keep the Python version the same:** First, install `networkx` with version 3.0 before installing PyTorch.

```bash
pip install networkx==3.0
# Then proceed with the installation of PyTorch.
```

**2. Dependency Not Found**

If you encounter errors similar to:

```bash
ERROR: Could not find a version that satisfies the requirement librosa==0.9.1 (from versions: none)
ERROR: No matching distribution found for librosa==0.9.1
# Key characteristics of the error:
No matching distribution found for xxxxx
Could not find a version that satisfies the requirement xxxx
```

**Solution:** Change the installation source. Add a download source when manually installing the dependency.

Use the command `pip install [package_name] -i [source_url]`. For example, to download `librosa` version 0.9.1 from the Alibaba source, use the following command:

```bash
pip install librosa==0.9.1 -i http://mirrors.aliyun.com/pypi/simple
```

**3. Certain dependencies cannot be installed due to a high pip version**

On June 21, 2024, pip was updated to version 24.1. Simply using `pip install --upgrade pip` will update pip to version 24.1. However, some dependencies require pip 23.0 to be installed, necessitating a manual downgrade of the pip version. It is currently known that hydra-core, omegaconf, and fastapi are affected by this. The specific error encountered during installation is as follows:

```bash
Please use pip<24.1 if you need to use this version.
INFO: pip is looking at multiple versions of hydra-core to determine which version is compatible with other requirements. This could take a while.
ERROR: Cannot install -r requirements.txt (line 20) and fairseq because these package versions have conflicting dependencies.

The conflict is caused by:
    fairseq 0.12.2 depends on omegaconf<2.1
    hydra-core 1.0.7 depends on omegaconf<2.1 and >=2.0.5

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip to attempt to solve the dependency conflict
```

The solution is to limit the pip version before installing dependencies as described in [1.5 Installation of Other Dependencies](#15-installation-of-other-dependencies). Use the following command to limit the pip version:

```bash
pip install --upgrade pip==23.3.2 wheel setuptools
```

After running this command, proceed with the installation of other dependencies.

## Common Errors During Dataset Preprocessing and Model Training

**1. Error: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0 in position xx`**

- Ensure that dataset filenames do not contain non-Western characters such as Chinese or Japanese, especially Chinese punctuation marks like brackets, commas, colons, semicolons, quotes, etc. After renaming, **reprocess the dataset** and then proceed with training.

**2. Error: `The expand size of the tensor (768) must match the existing size (256) at non-singleton dimension 0.`**

- Delete all contents under `dataset/44k` and redo the preprocessing steps.

**3.Error: `RuntimeError: DataLoader worker (pid(s) 13920) exited unexpectedly`**

```bash
raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
RuntimeError: DataLoader worker (pid(s) 13920) exited unexpectedly
```

- Reduce the `batchsize` value, increase virtual memory, and restart the computer to clear GPU memory until the `batchsize` value and virtual memory are appropriate and do not cause errors.

**4. Error: `torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with exit code 3221225477`**

- Increase virtual memory and reduce the `batchsize` value until the `batchsize` value and virtual memory are appropriate and do not cause errors.

**5. Error: `AssertionError: CPU training is not allowed.`**

- **No solution:** Training without an NVIDIA GPU is not supported. For beginners, the straightforward answer is that training without an NVIDIA GPU is not feasible.

**6. Error: `FileNotFoundError: No such file or directory: 'pretrain/rmvpe.pt'`**

- If you run `python preprocess_hubert_f0.py --f0_predictor rmvpe --use_diff` and encounter `FileNotFoundError: No such file or directory: 'pretrain/rmvpe.pt'`, this is because the official documentation updated the rmvpe preprocessor for F0 processing. Refer to the tutorial documentation [#2.2.3](#223-optional-items-choose-as-needed) to download the preprocessing model `rmvpe.pt` and place it in the corresponding directory.

**7. Error: "Page file is too small to complete the operation."**

- **Solution:** Increase the virtual memory. You can find detailed instructions online for your specific operating system.

## Errors When Using WebUI\*\*

**1. Errors When Starting or Loading Models in WebUI**

- **Error When Starting WebUI:** `ImportError: cannot import name 'Schema' from 'pydantic'`
- **Error When Loading Models in WebUI:** `AttributeError("'Dropdown' object has no attribute 'update'")`
- **Errors Related to Dependencies:** If the error involves `fastapi`, `gradio`, or `pydantic`.

**Solution:**

- Some dependencies need specific versions. After installing `requirements_win.txt`, enter the following commands in the command prompt to update the packages:

```bash
pip install --upgrade fastapi==0.84.0
pip install --upgrade gradio==3.41.2
pip install --upgrade pydantic==1.10.12
```

**2. Error: `Given groups=1, weight of size [xxx, 256, xxx], expected input[xxx, 768, xxx] to have 256 channels, but got 768 channels instead`**

or **Error: Encoder and model dimensions do not match in the configuration file**

- **Cause:** A v1 branch model is using a `vec768` configuration file, or vice versa.
- **Solution:** Check the `ssl_dim` setting in your configuration file. If `ssl_dim` is 256, the `speech_encoder` should be `vec256|9`. If it is 768, it should be `vec768|12`.
- For detailed instructions, refer to [#2.1](#21-issues-regarding-compatibility-with-the-40-model).

**3. Error: `'HParams' object has no attribute 'xxx'`**

- **Cause:** Usually, this indicates that the timbre cannot be found and the configuration file does not match the model.
- **Solution:** Open the configuration file and scroll to the bottom to check if it includes the timbre you trained.

# Acknowledgements

We would like to extend our heartfelt thanks to the following individuals and organizations whose contributions and resources have made this project possible:

- **so-vits-svc** | [so-vits-svc GitHub Repository](https://github.com/svc-develop-team/so-vits-svc)
- **GPT-SoVITS** | [GPT-SoVITS GitHub Repository](https://github.com/RVC-Boss/GPT-SoVITS)
- **SingingVocoders** | [SingingVocoders GitHub Repository](https://github.com/openvpi/SingingVocoders)
- **MoeVoiceStudio** | [MoeVoiceStudio GitHub Repository](https://github.com/NaruseMioShirakana/MoeVoiceStudio)
- **OpenVPI** | [OpenVPI GitHub Organization](https://github.com/openvpi) | [Vocoders GitHub Repository](https://github.com/openvpi/vocoders)
- **Up 主 [infinite_loop]** | [Bilibili Profile](https://space.bilibili.com/286311429) | [Related Video](https://www.bilibili.com/video/BV1Bd4y1W7BN) | [Related Column](https://www.bilibili.com/read/cv21425662)
- **Up 主 [羽毛布団]** | [Bilibili Profile](https://space.bilibili.com/3493141443250876) | [Error Resolution Guide](https://www.bilibili.com/read/cv22206231) | [Common Error Solutions](https://www.yuque.com/umoubuton/ueupp5/ieinf8qmpzswpsvr)
- **All Contributors of Training Audio Samples**
- **You** - For your interest, support, and contributions.
