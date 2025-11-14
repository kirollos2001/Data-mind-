# Analyst AI Agent

A structured Python project with a clean architecture and development tools.

## Project Structure

```
Analyst_AI_Agent/
├── .github/
│   └── copilot-instructions.md
├── src/
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

Run the main application:
```bash
python src/main.py
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

Format code with Black:
```bash
black src/ tests/
```

### Linting

Check code with flake8:
```bash
flake8 src/ tests/
```

### Type Checking

Run mypy for type checking:
```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License.
