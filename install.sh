#!/bin/bash

# DiscoSeqSampler Installation Script
echo "Installing DiscoSeqSampler..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

required_version="3.8"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python version meets requirements (>= 3.8)"
else
    echo "✗ Python version $python_version is too old. Please use Python 3.8 or newer."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing DiscoSeqSampler in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov black isort flake8 mypy pre-commit

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Run tests to verify installation
echo "Running tests to verify installation..."
python -m pytest discoseqsampler/tests/ -v

echo "✓ Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the CLI, run:"
echo "  discoseq --help"
echo ""
echo "To run tests:"
echo "  pytest discoseqsampler/tests/"
