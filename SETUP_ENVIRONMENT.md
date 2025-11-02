# How to Set Environment Variables

## Option 1: Azure OpenAI (Recommended)

### Windows PowerShell

1. **Current session only** (temporary):
```powershell
$env:AZURE_OPENAI_API_KEY = "your-api-key-here"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
$env:AZURE_OPENAI_DEPLOYMENT = "your-deployment-name"
```

2. **Persistent** (permanent for your user):
```powershell
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_KEY", "your-api-key-here", "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com/", "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT", "your-deployment-name", "User")
```

After setting persistent variables, restart your terminal/IDE.

### Windows Command Prompt (CMD)

1. **Current session only**:
```cmd
set AZURE_OPENAI_API_KEY=your-api-key-here
set AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
set AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

2. **Persistent**:
```cmd
setx AZURE_OPENAI_API_KEY "your-api-key-here"
setx AZURE_OPENAI_ENDPOINT "https://your-resource-name.openai.azure.com/"
setx AZURE_OPENAI_DEPLOYMENT "your-deployment-name"
```

After using `setx`, restart your terminal/IDE.

### Linux/Mac

```bash
export AZURE_OPENAI_API_KEY="your-api-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

For permanent setup, add to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export AZURE_OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"' >> ~/.bashrc
echo 'export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"' >> ~/.bashrc
source ~/.bashrc
```

## Option 2: Standard OpenAI

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
```

### Windows CMD

```cmd
set OPENAI_API_KEY=your-openai-api-key
```

### Linux/Mac

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Verify Environment Variables

### Windows PowerShell
```powershell
echo $env:AZURE_OPENAI_API_KEY
echo $env:AZURE_OPENAI_ENDPOINT
echo $env:AZURE_OPENAI_DEPLOYMENT
```

### Windows CMD
```cmd
echo %AZURE_OPENAI_API_KEY%
echo %AZURE_OPENAI_ENDPOINT%
echo %AZURE_OPENAI_DEPLOYMENT%
```

### Linux/Mac
```bash
echo $AZURE_OPENAI_API_KEY
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_DEPLOYMENT
```

## Quick Setup Script (PowerShell)

Save this as `setup_env.ps1`:

```powershell
# Azure OpenAI Setup
Write-Host "Setting up Azure OpenAI environment variables..." -ForegroundColor Green

# You can edit these values
$AZURE_OPENAI_API_KEY = "your-api-key-here"
$AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
$AZURE_OPENAI_DEPLOYMENT = "your-deployment-name"

# Set for current session
$env:AZURE_OPENAI_API_KEY = $AZURE_OPENAI_API_KEY
$env:AZURE_OPENAI_ENDPOINT = $AZURE_OPENAI_ENDPOINT
$env:AZURE_OPENAI_DEPLOYMENT = $AZURE_OPENAI_DEPLOYMENT

Write-Host "Environment variables set for this session!" -ForegroundColor Green
Write-Host "To verify, run: python example_usage.py" -ForegroundColor Yellow
```

Run it with:
```powershell
.\setup_env.ps1
```

## Using a .env File (Alternative)

You can also use a `.env` file with `python-dotenv`:

1. Create a `.env` file in your project root:
```
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

2. Install python-dotenv (already in requirements.txt):
```bash
pip install python-dotenv
```

3. Load it in your code:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Troubleshooting

### Variables not persisting?
- **PowerShell**: Use `[Environment]::SetEnvironmentVariable()` with "User" scope
- **CMD**: Use `setx` command
- Restart your terminal/IDE after setting persistent variables

### Still not working?
- Check for typos in variable names
- Make sure there are no extra spaces
- Verify the values are correct (especially the endpoint URL)
- Try setting them in the current session first to test

## Next Steps

After setting your environment variables:
1. Restart your terminal/IDE
2. Run: `python example_usage.py`
3. Or run: `python test_planner.py`

The planner agent will automatically detect and use your credentials!

