# Quick Start Guide - Planner Agent

## Overview
The Planner Agent is a LangChain-based system that automatically analyzes GitHub repositories and generates comprehensive installation guides.

## Files Created

1. **planner_agent.py** (593 lines)
   - Main implementation with PlannerAgent class
   - Supports 7 programming languages
   - GitHub API integration
   - Web search for best practices
   - LLM-powered guide generation

2. **requirements.txt**
   - All necessary dependencies listed
   - Install with: `pip install -r requirements.txt`

3. **example_usage.py**
   - Demonstration of the agent
   - Shows both API calls and full usage
   - No API key needed for API demo

4. **test_planner.py**
   - Simple test suite
   - Validates core functionality

5. **README.md**
   - Complete documentation
   - Usage examples
   - Features and architecture

## Quick Test

```bash
# 1. Install dependencies (already done)
python -m pip install -r requirements.txt

# 2. Run demo (no API key needed)
python example_usage.py

# 3. With OpenAI API key (for full functionality)
# Option A: Standard OpenAI
export OPENAI_API_KEY="your-key-here"

# Option B: Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="your-deployment"

# Run tests
python test_planner.py
```

## Core Features

✅ GitHub repository analysis  
✅ Multi-language support (Python, JS, Go, Rust, Java, C/C++)  
✅ Documentation reading (README, INSTALL, BUILD files)  
✅ Build system detection  
✅ Web search integration  
✅ LLM-generated installation guides  
✅ Structured output (InstallationGuide dataclass)  

## Supported Languages & Build Tools

- **Python**: pip, poetry, conda, setuptools
- **JavaScript**: npm, yarn, pnpm, bun
- **Go**: go, make
- **Rust**: cargo, rustc
- **Java**: maven, gradle, ant
- **C++**: cmake, make, bazel, ninja
- **C**: make, cmake, gcc, clang

## Next Steps

- Implement Builder Agent to execute the guides
- Add containerization support
- Batch processing for multiple repos
- Enhanced error handling

