# CRA Planner Agent

An intelligent agent that analyzes GitHub repositories and generates Dockerfiles using LangChain and Azure OpenAI.

## Features

- **Intelligent Repository Analysis**: Automatically detects project language and structure
- **Dockerfile Generation**: Creates optimized Dockerfiles based on project requirements
- **Smart Retry Mechanism**: Thread-safe iteration limiting prevents endless tool calls
- **Robust Git Cloning**: 4 fallback strategies with retry logic for reliable repository access
- **Docker Build Testing**: Validates generated Dockerfiles with error feedback
- **Iterative Refinement**: Analyzes build errors and refines Dockerfiles (up to 15 tool calls)
- **Parallel Testing**: Test multiple repositories concurrently with 4 workers

## Architecture

### Feedback Loops

1. **Chat History Feedback** (`run_agent.py` lines 552-684):
   - Query 1: Initial repository search and language detection
   - Query 2: Deep dive into project structure and dependencies
   - Query 3: Dependency resolution and package analysis
   - Query 4: Dockerfile generation using accumulated knowledge
   - Smart retry: Detects stuck behavior, forces answer with max_iterations=1

2. **Dockerfile Refinement Feedback** (`parallel_empirical_test.py` lines 280-560):
   - Build Dockerfile and capture errors
   - Detect error type: IMAGE_PULL, DEPENDENCY, BUILD
   - Generate targeted refinement query with error context
   - Allow up to 15 tool calls for thorough investigation
   - Track tool usage to ensure agent explores codebase
   - Iterate until build succeeds or max refinements reached

## Setup

### Prerequisites

- Python 3.11+
- Docker
- Azure OpenAI API access (gpt-5-nano deployment)

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI credentials:
# AZURE_OPENAI_API_KEY=your_key
# AZURE_OPENAI_ENDPOINT=your_endpoint
# AZURE_OPENAI_API_VERSION=2024-02-15-preview
# AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-nano
```

## Usage

### Single Repository Analysis

```bash
python src/run_agent.py <repository_url>
```

### Parallel Testing

```bash
# Test repositories from failed.txt (one URL per line)
python src/parallel_empirical_test.py failed.txt

# Specify number of worker threads (default: 4)
python src/parallel_empirical_test.py failed.txt --workers 8
```

### Docker Build

```bash
# Build the agent container
docker build -t cra-planner-agent .

# Run the agent
docker run -v $(pwd)/results:/app/results \
  -e AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY \
  -e AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT \
  cra-planner-agent
```

## Configuration

### Agent Parameters

- **max_iterations**: Default 10, increased to 15 for refinement queries
- **max_execution_time**: 1800 seconds (30 minutes)
- **iteration_limit**: 5 for Query 4 smart retry

### Git Clone Strategies

1. Standard full clone (30min timeout)
2. Shallow clone depth=1 (15min timeout)
3. HTTP/1.1 clone (fixes HTTP/2 stream errors)
4. Single-branch clone (minimal transfer)

Each strategy retried 3 times with exponential backoff.

## Output

Results are saved to timestamped directories:

```
results_YYYYMMDD_HHMMSS/
├── <repo_name>/
│   ├── Dockerfile            # Generated Dockerfile
│   ├── analysis_report.txt   # Detailed analysis
│   ├── metrics.json          # Token usage and timing
│   └── logs.txt              # Agent execution logs
```

## Testing

```bash
# Run unit tests
python tests/test_agent.py

# Test Azure OpenAI connection
python tests/test_azure_openai.py
```

## Troubleshooting

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Git Clone Failures

The agent automatically retries with different strategies. For persistent failures:
- Check network connectivity
- Verify repository is public or credentials are correct
- Try manually cloning to test access

### Docker Build Failures

Check logs in `results/*/logs.txt` for detailed error messages. Common issues:
- Missing dependencies in requirements detection
- Incorrect base image selection
- Network issues during image pull

## Development

### Key Components

- `planner_agent.py`: Core agent with tools (SearchWeb, ReadFile, DockerImageSearch, etc.)
- `run_agent.py`: Repository analysis workflow orchestration
- `parallel_empirical_test.py`: Parallel testing framework
- `empirical_test.py`: Docker build testing and error categorization

### Thread Safety

All parallel operations use thread-safe mechanisms:
- `ThreadAwareStdout` for logging
- `_invoke_agent_with_iteration_limit()` for iteration control
- Isolated agent instances per test

## License

MIT License - See LICENSE file for details
