# Azure OpenAI Configuration Example

## Environment Variables

Set these environment variables to use Azure OpenAI:

```bash
# Azure OpenAI API Key
export AZURE_OPENAI_API_KEY="your-api-key-here"

# Azure OpenAI Endpoint
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"

# Deployment Name (the name you gave your model deployment)
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

## Windows PowerShell

```powershell
$env:AZURE_OPENAI_API_KEY="your-api-key-here"
$env:AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
$env:AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

## Windows Command Prompt

```cmd
set AZURE_OPENAI_API_KEY=your-api-key-here
set AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
set AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

## Usage

Once the environment variables are set, the PlannerAgent will automatically use Azure OpenAI:

```python
from planner_agent import PlannerAgent

# Automatically detects Azure credentials from environment
planner = PlannerAgent()

# Or explicitly force Azure mode
planner = PlannerAgent(use_azure=True)

# Generate installation guide
guide = planner.generate_installation_guide("https://github.com/psf/requests")
```

## API Version

The PlannerAgent uses API version `2024-02-15-preview`. If you need a different version, you can modify line 95 in `planner_agent.py`.

