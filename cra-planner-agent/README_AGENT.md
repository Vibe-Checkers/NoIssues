# LangChain Agent with Azure-Hosted OpenRouter

A simple LangChain agent implementation using Azure-hosted OpenRouter GPT model.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=your_azure_openrouter_endpoint_here
OPENROUTER_MODEL=your_model_name_here
```

**Example:**
```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://your-azure-endpoint.openai.azure.com/
OPENROUTER_MODEL=gpt-4
```

### 3. Run the Agent

```bash
python langchain_agent.py
```

## Features

The agent comes with three built-in tools:

1. **GetCurrentTime**: Returns the current date and time
2. **Calculator**: Performs basic arithmetic calculations
3. **SearchInfo**: Simulates information search (can be replaced with actual API)

## Customization

### Adding New Tools

To add new tools, define a function and add it to the `tools` list in `langchain_agent.py`:

```python
def my_custom_tool(query: str) -> str:
    """Description of what the tool does."""
    # Your implementation
    return result

tools.append(
    Tool(
        name="MyCustomTool",
        func=my_custom_tool,
        description="Description for the agent"
    )
)
```

### Modifying the Agent

You can adjust the agent's behavior by modifying:
- `temperature`: Controls randomness (0.0-1.0)
- `max_iterations`: Maximum reasoning steps
- `verbose`: Set to False to reduce output

## Usage Example

```python
from langchain_agent import create_langchain_agent

# Create the agent
agent = create_langchain_agent()

# Run a query
result = agent.invoke({"input": "What is 15 * 23?"})
print(result['output'])
```

## Troubleshooting

- Ensure your `.env` file is in the project root
- Verify your OpenRouter API key is valid
- Check that the model name matches your Azure deployment
- Make sure the base URL includes the full Azure endpoint
