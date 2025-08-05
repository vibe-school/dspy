---
sidebar_position: 2
---

# Environment Setup

Let's get your environment ready for the DSPy tutorial. This guide will walk you through all the necessary steps.

## System Requirements

- **Python 3.10 or higher** (required)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 8GB RAM recommended
- **Internet Connection**: Required for API calls

## Installation Options

### Option 1: Using pip (Traditional)

```bash
# Create a virtual environment
python -m venv dspy-env

# Activate it
# On Windows:
dspy-env\Scripts\activate
# On macOS/Linux:
source dspy-env/bin/activate

# Install DSPy and dependencies
pip install dspy-ai openai google-generativeai
```

### Option 2: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust.

```bash
# Install uv
pip install uv

# Clone the tutorial repository
git clone https://github.com/dspy-tutorial/dspy-den
cd dspy-den/context-engineering-dspy

# Install dependencies
uv sync
```

## API Keys Configuration

You'll need API keys for the LLM providers you want to use. Here are the main options:

### OpenAI

1. Get your API key from [platform.openai.com](https://platform.openai.com)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Google Gemini

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

### Tavily (For Web Search - Levels 4 & 5)

1. Sign up at [tavily.com](https://tavily.com)
2. Get your API key from the dashboard
3. Set the environment variable:
   ```bash
   export TAVILY_API_KEY="your-api-key-here"
   ```

## Environment Management

### Option 1: Using direnv (Recommended)

```bash
# Install direnv (if not already installed)
# macOS: brew install direnv
# Ubuntu: sudo apt install direnv

# Create .envrc file in your project directory
cat > .envrc << EOF
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export TAVILY_API_KEY="your-tavily-key"
EOF

# Allow direnv to load the file
direnv allow
```

### Option 2: Using .env file with python-dotenv

```bash
# Install python-dotenv
pip install python-dotenv

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
EOF
```

Then in your Python scripts:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Option 3: Export in Shell Profile

Add to your `~/.bashrc`, `~/.zshrc`, or equivalent:
```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export TAVILY_API_KEY="your-tavily-key"
```

## Verify Installation

Create a test file `test_setup.py`:

```python
import dspy
import os

# Check DSPy version
print(f"DSPy version: {dspy.__version__}")

# Check environment variables
api_keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")
}

for key, value in api_keys.items():
    if value:
        print(f"✅ {key} is set")
    else:
        print(f"❌ {key} is not set")

# Test basic DSPy functionality
try:
    # Configure with OpenAI
    if api_keys["OPENAI_API_KEY"]:
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
        print("✅ DSPy configured with OpenAI")
    elif api_keys["GEMINI_API_KEY"]:
        dspy.configure(lm=dspy.LM("gemini/gemini-2.0-flash"))
        print("✅ DSPy configured with Gemini")
    else:
        print("❌ No LLM API key found")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

Run the test:
```bash
python test_setup.py
```

## Additional Dependencies

For specific tutorial levels, you may need additional packages:

### Level 3 (Evaluation)
```bash
pip install mlflow pandas seaborn matplotlib
```

### Level 5 (RAG)
```bash
pip install annoy rank-bm25 transformers torch
```

## Troubleshooting

### Common Issues

1. **Python Version Error**
   ```
   ERROR: This script requires Python 3.10 or higher
   ```
   Solution: Update Python or use pyenv to manage versions

2. **Import Error for DSPy**
   ```
   ModuleNotFoundError: No module named 'dspy'
   ```
   Solution: Install with `pip install dspy-ai` (not just `dspy`)

3. **API Key Errors**
   ```
   OpenAI API key not found
   ```
   Solution: Ensure environment variables are properly set and loaded

4. **SSL Certificate Errors**
   ```
   SSL: CERTIFICATE_VERIFY_FAILED
   ```
   Solution: Update certificates or use corporate proxy settings

## Next Steps

Once your environment is set up and verified:

1. ✅ All API keys are configured
2. ✅ DSPy is installed and working
3. ✅ Test script runs successfully

You're ready to start with [Module 1: DSPy Fundamentals](/tutorial/module1/fundamentals)!

---

<div style={{textAlign: 'center', marginTop: '2rem'}}>
  <a className="button button--secondary button--lg" href="/tutorial/intro">
    ← Back to Introduction
  </a>
  <span style={{margin: '0 1rem'}}></span>
  <a className="button button--primary button--lg" href="/tutorial/module1/fundamentals">
    Continue to Module 1 →
  </a>
</div>