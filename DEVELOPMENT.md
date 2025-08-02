# 开发环境设置指南

## 快速开始

### 1. 安装开发依赖

```bash
# 克隆仓库
git clone https://github.com/lifeiteng/DiscoSeqSampler.git
cd DiscoSeqSampler

# 安装开发依赖
pip install -e .[dev]
```

### 2. 设置 Pre-commit 钩子

```bash
# 使用 Makefile（推荐）
make setup-dev

# 或者手动安装
pre-commit install
pre-commit install --hook-type commit-msg
```

## 开发工具配置

### 代码格式化和检查工具

项目配置了以下工具来确保代码质量：

- **Black**: Python 代码格式化
- **Ruff**: 快速的 Python linter（替代 flake8、isort 等）
- **MyPy**: 静态类型检查
- **Bandit**: 安全漏洞检查
- **Pytest**: 测试框架
- **Pre-commit**: Git 钩子自动化

### 常用命令

```bash
# 设置开发环境
make setup-dev

# 安装和运行 pre-commit 钩子
make pre-commit

# 直接使用工具命令
black .              # 格式化代码
ruff check .         # 运行 linting
ruff format .        # 格式化代码
mypy discoss         # 类型检查
pytest               # 运行测试
pytest --cov=discoss # 运行测试并生成覆盖率
```

### IDE 配置

#### VS Code

推荐安装以下扩展：

- Python
- Black Formatter
- Ruff
- Mypy Type Checker
- Pre-commit

项目已配置 `.editorconfig` 文件来确保编辑器设置一致。

#### PyCharm

- 启用 Black 作为代码格式化工具
- 配置 Ruff 作为外部工具
- 启用 MyPy 插件

## Git 工作流

### Pre-commit 钩子

每次提交时会自动运行：

1. 代码格式化（Black, Ruff）
2. 代码检查（Ruff linting）
3. 类型检查（MyPy）
4. 安全检查（Bandit）
5. 基础检查（尾随空格、文件结尾等）
6. 提交信息格式检查

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_init.py

# 运行带覆盖率的测试
pytest --cov=discoss

# 生成 HTML 覆盖率报告
pytest --cov=discoss --cov-report=html
```

### 测试组织

- `tests/` - 测试文件目录
- `tests/conftest.py` - 测试配置和夹具
- 测试文件命名：`test_*.py`
- 测试函数命名：`test_*`

### 测试标记

使用 pytest markers 来分类测试：

```python
@pytest.mark.unit
def test_unit_function():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass
```

运行特定类型的测试：
```bash
pytest -m unit          # 只运行单元测试
pytest -m "not slow"    # 跳过慢速测试
```

## CI/CD

项目配置了 GitHub Actions 工作流：

- **CI 流水线**：在每次 push 和 PR 时运行
  - 多 Python 版本测试（3.10, 3.11, 3.12）
  - 代码质量检查
  - 安全检查
  - 测试覆盖率上传到 Codecov

- **构建流水线**：在 main 分支更新时运行
  - 构建包
  - 验证包完整性

## 发布流程

1. 更新版本号在 `discoss/__init__.py`
2. 更新 CHANGELOG.md
3. 创建 Git tag
4. GitHub Actions 会自动构建和测试

```bash
# 手动构建和检查
python -m build
twine check dist/*

# 上传到 PyPI（需要配置凭据）
twine upload dist/*
```

## 故障排除

### 常见问题

1. **Pre-commit 失败**
   ```bash
   # 更新 pre-commit 钩子
   pre-commit autoupdate
   pre-commit run --all-files
   ```

2. **MyPy 类型错误**
   ```bash
   # 安装类型存根
   pip install types-requests types-setuptools
   ```

3. **测试失败**
   ```bash
   # 清理缓存
   rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
   pytest --cache-clear
   ```

### 获取帮助

- 查看 Makefile：`make help`
- 检查工具配置：`pyproject.toml`
- 查看 CI 日志：GitHub Actions 页面
