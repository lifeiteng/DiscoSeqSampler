# DiscoSeqSampler 示例

本目录包含了使用 DiscoSeqSampler 处理多模态数据的完整示例。

## 概述

DiscoSeqSampler 是一个基于 Lhotse 的分布式序列采样框架，专门为处理音频、图像和视频等多模态数据而设计。通过智能的序列长度管理和分布式协调机制，它能够显著提升大规模 Transformer 模型的训练效率。

## 文件说明

- `01-audio-image-video-cuts.ipynb` - 完整的多模态数据处理示例，展示如何创建音频、图像和视频的 cuts 文件
- `audio_cuts.jsonl.gz` - 示例音频 cuts 数据
- `image_cuts.jsonl.gz` - 示例图像 cuts 数据
- `video_cuts.jsonl.gz` - 示例视频 cuts 数据

## 快速开始

### 1. 环境准备

确保您已安装必要的依赖：

```bash
pip install discoss
conda install ffmpeg  # 用于视频处理
```

### 2. 数据准备

示例使用 LibriTTS 数据集进行演示：

```bash
# 下载并准备 LibriTTS 数据集
lhotse download libritts -p dev-clean tests/data
lhotse prepare libritts -p dev-clean tests/data/LibriTTS tests/data/manifests/libritts
```

### 3. 创建 Cuts 文件

Cuts 是 Lhotse 中的核心概念，表示音频、视频或其他数据的片段。运行示例 notebook：

```bash
jupyter notebook 01-audio-image-video-cuts.ipynb
```

或者使用 Lhotse CLI 快速创建音频 cuts：

```bash
lhotse cut simple --force-eager \
    -r tests/data/manifests/libritts/libritts_recordings_dev-clean.jsonl.gz \
    -s tests/data/manifests/libritts/libritts_supervisions_dev-clean.jsonl.gz \
    examples/audio_cuts.jsonl.gz
```

## 主要功能展示

### 音频处理
- 从音频文件创建 cuts
- 添加转录文本和监督信息
- 处理不同格式的音频数据

### 图像处理
- 从图像文件创建 cuts
- 附加图像数据到 cuts 对象
- 添加图像描述文本

### 视频处理
- 处理带音频轨道的视频文件
- 处理无音频轨道的视频文件
- 提取视频元数据（帧率、分辨率等）
- 添加视频描述文本

## 数据格式

所有生成的 cuts 文件都采用 JSONL（JSON Lines）格式，并使用 gzip 压缩。每行包含一个 cut 对象的完整信息，包括：

- 元数据（ID、时长、采样率等）
- 监督信息（文本标注、时间戳等）
- 自定义字段（图像数据、视频信息等）

## 下一步

创建 cuts 文件后，您可以：

1. 使用 DiscoSeqSampler 的各种采样器进行智能采样
2. 在分布式训练中实现高效的数据加载
3. 根据序列长度进行动态批处理
4. 实现多模态数据的协调训练

更多高级用法请参考主项目文档和测试用例。
