# DiscoSeqSampler

**Distributed Coordinated Sequenced Sampler for Speech Data**

DiscoSeqSampler 是一个基于 [Lhotse](https://github.com/lhotse-speech/lhotse) 的分布式协调序列采样器，专为语音数据的高效分布式训练而设计。

## 特性

- 🚀 **多种采样策略**: 支持顺序采样、分桶采样、随机采样等
- 🔄 **分布式协调**: 完整的多GPU分布式训练支持
- 📦 **动态批处理**: 基于时长或帧数的智能批处理
- 🎯 **内存优化**: 二次时长估计和缓冲区管理
- 🔧 **容错机制**: 状态管理和检查点支持
- ⚡ **高性能**: 高效的数据加载和预取

## 安装

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/feiteng/DiscoSeqSampler.git
cd DiscoSeqSampler
```

### 手动安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 快速开始

### 基本用法

```python
from discoseqsampler import DiscoSeqSampler, SamplerConfig, SamplingStrategy
from lhotse import CutSet
from torch.utils.data import DataLoader

# 加载你的数据
cuts = CutSet.from_file("path/to/your/cuts.jsonl.gz")

# 配置采样器
config = SamplerConfig(
    strategy=SamplingStrategy.BUCKETED,  # 分桶采样
    max_duration=30.0,                   # 每批最大30秒
    world_size=1,                        # 单GPU
    rank=0,
    shuffle=True,
    num_buckets=10
)

# 创建采样器
sampler = DiscoSeqSampler(cuts, config)

# 使用DataLoader
dataloader = DataLoader(
    sampler,
    batch_size=None,  # 由采样器控制批大小
    num_workers=4
)

# 训练循环
for epoch in range(10):
    sampler.set_epoch(epoch)  # 设置epoch确保确定性采样
    
    for batch in dataloader:
        # 你的训练代码
        pass
```

### 分布式训练

```python
import torch.distributed as dist
from discoseqsampler import create_dataloader

# 初始化分布式
dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 配置分布式采样
config = SamplerConfig(
    strategy=SamplingStrategy.BUCKETED,
    max_duration=20.0,
    world_size=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,  # 分布式训练推荐
)

# 创建数据加载器
dataloader = create_dataloader(cuts, config)

# 分布式训练循环
for epoch in range(epochs):
    dataloader.dataset.set_epoch(epoch)
    
    for batch in dataloader:
        # 你的分布式训练代码
        pass
```

## 采样策略

### 1. 顺序采样 (Sequential)
按原始顺序处理数据，适用于：
- 可重现的训练
- 评估场景
- 调试分析

### 2. 分桶采样 (Bucketed)
根据相似特征分组，优势：
- 减少批内padding
- 提高内存利用率
- 更均匀的处理时间

### 3. 随机采样 (Random)
完全随机采样，适用于：
- 标准训练场景
- 数据增强

## 配置选项

### SamplerConfig 参数

```python
config = SamplerConfig(
    # 采样策略
    strategy=SamplingStrategy.BUCKETED,
    
    # 批处理配置
    max_duration=30.0,      # 最大批持续时间（秒）
    max_cuts=None,          # 最大切片数量
    
    # 分布式配置
    world_size=1,           # 工作进程数
    rank=0,                 # 当前进程rank
    
    # 随机化
    seed=42,                # 随机种子
    shuffle=True,           # 是否打乱
    drop_last=False,        # 是否丢弃最后不完整批
    
    # 分桶配置（仅用于BUCKETED策略）
    bucket_method="duration",  # 分桶方法: "duration", "num_frames", "num_features"
    num_buckets=10,           # 桶数量
    
    # 性能配置
    buffer_size=10000,        # 缓冲区大小
    quadratic_duration=False, # 二次时长估计
    num_workers=0,           # 数据加载工作进程数
    pin_memory=False,        # 内存固定
    prefetch_factor=2,       # 预取因子
)
```

## CLI 工具

DiscoSeqSampler 提供了命令行工具用于分析和基准测试：

```bash
# 分析采样行为
discoseq analyze path/to/cuts.jsonl.gz --strategy bucketed --max-duration 30.0 --output stats.json

# 性能基准测试
discoseq benchmark path/to/cuts.jsonl.gz config.json --epochs 3
```

## 示例

查看 `examples/` 目录获取更多使用示例：

- `basic_usage.py` - 基本用法示例
- `distributed_training.py` - 分布式训练示例
- `advanced_usage.py` - 高级功能示例

## 开发

### 运行测试

```bash
# 运行所有测试
pytest discoseqsampler/tests/

# 运行特定测试
pytest discoseqsampler/tests/test_sampler.py -v

# 运行测试并生成覆盖率报告
pytest discoseqsampler/tests/ --cov=discoseqsampler --cov-report=html
```

### 代码格式化

```bash
# 格式化代码
black discoseqsampler/
isort discoseqsampler/

# 类型检查
mypy discoseqsampler/

# 代码检查
flake8 discoseqsampler/
```

## 许可证

本项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献

欢迎贡献！请查看贡献指南了解如何参与项目开发。

## 致谢

- [Lhotse](https://github.com/lhotse-speech/lhotse) - 优秀的语音数据处理工具包
- PyTorch团队 - 分布式训练框架
