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

