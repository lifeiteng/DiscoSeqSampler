# DiscoSeqSampler

[![CI](https://github.com/lifeiteng/DiscoSeqSampler/actions/workflows/ci.yml/badge.svg)](https://github.com/lifeiteng/DiscoSeqSampler/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lifeiteng/DiscoSeqSampler/branch/main/graph/badge.svg)](https://codecov.io/gh/lifeiteng/DiscoSeqSampler)
[![PyPI version](https://badge.fury.io/py/discoss.svg)](https://badge.fury.io/py/discoss)
[![Python version](https://img.shields.io/pypi/pyversions/discoss.svg)](https://pypi.org/project/discoss/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Distributed Coordinated Sequence Sampler - 一个高效的分布式序列采样框架。

## 特性

- 🚀 **高性能**: 优化的分布式采样算法
- 🔄 **协调机制**: 智能的序列协调和同步
- 📊 **可扩展**: 支持大规模分布式部署
- 🛠️ **易用性**: 简洁的 API 设计
- 🔧 **可配置**: 灵活的配置选项

## 安装

### 从 PyPI 安装

```bash
pip install discoss
```

### 从源码安装

```bash
git clone https://github.com/lifeiteng/DiscoSeqSampler.git
cd DiscoSeqSampler
pip install -e .
```

## 快速开始

```python
import discoss

# TODO: 添加使用示例
```

## 开发

查看 [DEVELOPMENT.md](DEVELOPMENT.md) 获取详细的开发指南。

### 快速设置

```bash
# 克隆仓库
git clone https://github.com/lifeiteng/DiscoSeqSampler.git
cd DiscoSeqSampler

# 安装开发依赖
pip install -e .[dev]

# 设置 pre-commit 钩子
make setup-dev
```

### 运行测试

```bash
make test
```

## 贡献

欢迎贡献！请查看 [DEVELOPMENT.md](DEVELOPMENT.md) 了解如何设置开发环境。

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 引用

如果您在研究中使用了 DiscoSeqSampler，请引用：

```bibtex
@software{discoss2024,
  title={DiscoSeqSampler: Distributed Coordinated Sequence Sampler},
  author={Li, Feiteng},
  year={2025},
  url={https://github.com/lifeiteng/DiscoSeqSampler}
}
```
