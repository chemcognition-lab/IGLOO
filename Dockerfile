# Multi-stage Dockerfile for IGLOO
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements_dev.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Development stage
FROM base AS development

# Install development dependencies
RUN pip install -r requirements_dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 igloo && \
    chown -R igloo:igloo /app
USER igloo

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base AS production

# Copy only necessary files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY pyproject.toml ./

# Install package
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 igloo && \
    chown -R igloo:igloo /app
USER igloo

# Default command for production
CMD ["python", "-m", "igloo"]

# Testing stage
FROM development AS testing

# Run tests
RUN pytest tests/ -v

# Copy test results
RUN coverage run -m pytest tests/ && \
    coverage report && \
    coverage html
