FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for prompts and results
RUN mkdir -p /app/prompts /app/results /app/.prompt-optimizer

# Set environment variables
ENV PYTHONUNBUFFERED=1

# CLI entrypoint
ENTRYPOINT ["prompt-optimizer"]
CMD ["--help"]
