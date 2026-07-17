# Use Python 3.11 slim as the base image for stability and smaller size
FROM python:3.11-slim

# Set environment variables to avoid writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HEADLESS_MODE=1

# Create and set working directory
WORKDIR /app

# Install system dependencies required for compilation and numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port for the Streamlit UI (optional)
EXPOSE 8501

# Command to run the terminal chat or headless agent
# To run headless model training, override this command with `python app.py`
CMD ["streamlit", "run", "terminal_chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
