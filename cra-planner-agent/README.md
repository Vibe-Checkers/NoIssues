# Planner Agent - How to Run

## Quick Start

### 1. Install Dependencies

```bash
cd cra-planner-agent
pip install -r requirements.txt
```

### 2. Set Up Azure OpenAI Credentials

Create a `.env` file in the `cra-planner-agent` directory:

```env
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

### 3. Run the Agent

#### Option A: Full Repository Analysis (Recommended)

This clones a GitHub repository and generates build instructions:

```bash
python run_agent.py https://github.com/psf/requests
```

**What it does:**
- Clones the repository to `./temp/`
- Analyzes the repository structure
- Extracts build configuration files
- Reads documentation
- Generates comprehensive build instructions
- Saves results to `{repo_name}_BUILD_INSTRUCTIONS.md`

#### Option B: Simple Test

Quick test to verify the agent works:

```bash
python test_agent.py
```

#### Option C: Test Azure OpenAI Connection

Verify your Azure OpenAI credentials:

```bash
python test_azure_openai.py
```

## Project Structure

```
cra-planner-agent/
├── planner_agent.py      # Core agent module
├── run_agent.py          # Main workflow script (clones & analyzes repos)
├── test_agent.py         # Simple test script
├── test_azure_openai.py  # Azure OpenAI connection test
├── requirements.txt      # Python dependencies
├── .env                  # Azure OpenAI credentials (create this)
└── README.md            # This file
```

## Requirements

- Python 3.8+
- Git (for cloning repositories)
- Azure OpenAI API credentials
- Internet connection

## Example Usage

```bash
# Analyze a Python project
python run_agent.py https://github.com/psf/requests

# Analyze a JavaScript project
python run_agent.py https://github.com/expressjs/express

# Analyze a Go project
python run_agent.py https://github.com/gin-gonic/gin
```

## Output

The agent will:
1. Clone the repository
2. Analyze it in 4 steps:
   - Step 1: Directory structure and project type
   - Step 2: Build configuration files
   - Step 3: Installation instructions from README
   - Step 4: Comprehensive build instructions
3. Save build instructions to a markdown file
4. Display tool usage and token usage statistics

## Troubleshooting

### "Azure OpenAI API key not provided"
- Make sure you have a `.env` file with `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_DEPLOYMENT`

### "Git is not installed"
- Install Git: https://git-scm.com/downloads

### "Module not found"
- Run: `pip install -r requirements.txt`

### Connection errors
- Verify your Azure OpenAI credentials are correct
- Check your internet connection
- Run `python test_azure_openai.py` to test the connection


