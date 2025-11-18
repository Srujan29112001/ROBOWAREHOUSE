# Contributing to RoboVLA

Thank you for your interest in contributing to RoboVLA! We welcome contributions from the community.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to team@robovla.ai.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Detailed steps to reproduce the problem
- Expected behavior vs. actual behavior
- Screenshots if applicable
- Environment details (OS, Python version, GPU, CUDA version)
- Any relevant log output

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- Detailed description of the proposed functionality
- Use cases and examples
- Any potential implementation approaches

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for any new functionality
4. **Ensure all tests pass** by running `pytest tests/`
5. **Update documentation** if you've changed APIs or added features
6. **Write clear commit messages** following conventional commits format
7. **Submit a pull request** with a clear description of changes

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA 12.1+ with cuDNN 8
- NVIDIA GPU (RTX 3060 or better recommended)

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/robo-vla.git
cd robo-vla

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=robo_vla tests/

# Run specific test file
pytest tests/test_perception.py -v

# Run specific test
pytest tests/test_perception.py::test_dino_detector -v
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pre-commit** hooks to enforce standards

```bash
# Format code
black robo_vla/

# Check linting
flake8 robo_vla/

# Type checking
mypy robo_vla/

# Run all pre-commit checks
pre-commit run --all-files
```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use type hints for function signatures
- Write docstrings for all public functions and classes (Google style)
- Keep functions focused and small
- Prefer composition over inheritance
- Use descriptive variable names

### Example Function

```python
def process_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 480),
) -> torch.Tensor:
    """
    Process and normalize an image for model input.

    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target size for resizing (height, width)

    Returns:
        Processed image tensor (C, H, W) normalized to [0, 1]

    Raises:
        ValueError: If image dimensions are invalid
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.ndim}D")

    # Implementation...
    return processed_tensor
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(perception): add DINO v2 quantization support

Implemented INT8 quantization for DINO v2 to reduce memory usage
from 2.8GB to 2.1GB while maintaining 99.5% accuracy.

Closes #123
```

```
fix(server): handle WebSocket disconnection gracefully

Added proper error handling for WebSocket disconnections to prevent
server crashes during unstable network conditions.
```

## Project Structure

```
robo-vla/
├── robo_vla/              # Main package
│   ├── perception/        # Vision components
│   ├── language/          # Language models
│   ├── knowledge/         # Knowledge graphs
│   ├── vla_model/         # Core VLA model
│   ├── rl/                # Reinforcement learning
│   ├── server/            # FastAPI server
│   ├── training/          # Training utilities
│   └── utils/             # Helper utilities
├── scripts/               # Training scripts
├── examples/              # Example code
├── tests/                 # Test suite
├── docs/                  # Documentation
└── robo_vla/deployment/   # Deployment configs
```

## Testing Guidelines

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Use fixtures for common test setup
- Mock external dependencies (models, databases)
- Test edge cases and error conditions

### Example Test

```python
import pytest
import torch
from robo_vla.perception import DinoV2Detector


@pytest.fixture
def detector():
    """Create detector instance for testing."""
    config = {"perception": {"dino": {"model_name": "facebook/dinov2-base"}}}
    return DinoV2Detector(config, device="cpu")


def test_dino_detector_output_shape(detector):
    """Test DINO detector output has correct shape."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 518, 518)

    features = detector(images)

    assert features.shape[0] == batch_size
    assert features.shape[1] == 768  # Feature dimension
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update configuration examples if needed
- Create tutorials for major features

## Performance Considerations

- Profile code for performance bottlenecks
- Consider memory efficiency (this runs on 12GB GPU)
- Use mixed precision training where applicable
- Optimize data loading pipelines
- Document any performance implications

## Questions?

- Open an issue for questions
- Join our discussions
- Email: team@robovla.ai

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Recognition

Contributors will be recognized in our CONTRIBUTORS.md file and release notes.

Thank you for contributing to RoboVLA!
