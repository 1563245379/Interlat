# Contributing to Interlat

We welcome contributions to the Interlat Framework! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic knowledge of PyTorch and transformers

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/XiaoDu-flying/Interlat.git
   cd hidden-state-training
   ```

2. **Create a Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in [GitHub Issues](https://github.com/XiaoDu-flying/Interlat/issues)
2. If not, create a new issue with:
   - Clear description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)

### Submitting Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Changes**
   - Write clean, readable code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/

   # Run specific test file
   pytest tests/test_models.py

   # Check code formatting
   black --check .
   isort --check-only .

   # Type checking
   mypy src/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   # or
   git commit -m "fix: fix bug description"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin your-branch-name
   ```

   Then create a pull request on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

## 📝 Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 88 characters (Black default)

### Code Formatting

```bash
# Format code
black .
isort .

# Check formatting
black --check .
isort --check-only .
```

### Documentation

- Use clear, descriptive docstrings
- Follow [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- Update README and documentation for new features
- Include type hints for function parameters and return values

### Example Docstring

```python
def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: List[int] = None
) -> Dict[str, torch.Tensor]:
    """Extract hidden states from specified model layers.

    Args:
        model: The PyTorch model to extract states from.
        input_ids: Input token IDs tensor of shape (batch_size, seq_len).
        layers: List of layer indices to extract. If None, extracts from all layers.

    Returns:
        Dictionary mapping layer names to hidden state tensors.

    Raises:
        ValueError: If layer indices are out of range.
    """
```

## 🧪 Testing Guidelines

### Writing Tests

- Write unit tests for all new functions/classes
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies

### Test Structure

```python
import pytest
from your_module import your_function

class TestYourFunction:
    """Tests for your_function."""

    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        # Arrange
        input_data = ...
        expected_output = ...

        # Act
        result = your_function(input_data)

        # Assert
        assert result == expected_output

    def test_edge_case(self):
        """Test handling of edge cases."""
        pass

    def test_error_handling(self):
        """Test proper error handling."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_models.py::TestCustomModel::test_forward_pass

# Run tests in parallel
pytest -n 4
```

## 📚 Documentation

### Adding Documentation

1. **API Documentation**: Use clear docstrings for all public functions/classes
2. **User Guides**: Add tutorials and examples in the `docs/` directory
3. **README Updates**: Update the main README for new features

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## 🎯 Types of Contributions

We welcome various types of contributions:

### 🐛 Bug Fixes
- Fix reported issues
- Improve error handling
- Performance optimizations

### ✨ New Features
- New model architectures
- Additional training strategies
- Data processing improvements
- Evaluation metrics

### 📖 Documentation
- Tutorial improvements
- API documentation
- Examples and demos
- Translation to other languages

### 🧪 Testing
- Unit tests
- Integration tests
- Performance benchmarks
- Edge case coverage

### 🔧 Infrastructure
- CI/CD improvements
- Docker configurations
- Package management
- Development tools

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Change is backwards compatible (or breaking change is clearly marked)
- [ ] Commit messages are clear and descriptive

### Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing
- [ ] Tests pass
- [ ] New tests added for new functionality

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Added to the AUTHORS file
- Mentioned in release notes for significant contributions

## ❓ Questions?

If you have questions about contributing:

1. Check existing [GitHub Discussions](https://github.com/XiaoDu-flying/Interlat/discussions)
2. Create a new discussion in the Q&A category
3. Join our community chat (if available)

---

Thank you for contributing to Interlat! 🎉