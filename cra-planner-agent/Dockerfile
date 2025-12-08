# Dockerfile

# 1. Use Python 3.10 slim
FROM python:3.10-slim

# 2. Install system tools (Git + Docker Client)
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory to the project root
WORKDIR /app

# 4. Install Python dependencies
# We copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the source code
# This copies the 'src' folder from your host to /app/src
COPY src/ ./src/

# 6. Create directories for results/logs so permissions don't crash
# (Optional but good practice)
RUN mkdir -p parallel_empirical_results analysis_reports logs

# 7. Set the PYTHONPATH
# This ensures Python can find modules in 'src' no matter where we run command from
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 8. Define the Entrypoint
# We run the script from the src folder
ENTRYPOINT ["python", "src/parallel_empirical_test.py"]