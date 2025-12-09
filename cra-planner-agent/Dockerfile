# Multi-stage build for CRA Planner Agent
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install git for repository cloning
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY analysis/ ./analysis/
COPY tests/ ./tests/
COPY *.txt ./

# Ensure scripts are executable
RUN chmod +x src/*.py

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "src/run_agent.py"]
