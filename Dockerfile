FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libpq-dev is needed for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
# Adding psycopg2-binary and schedule manually as they might not be in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for reports
RUN mkdir -p nightly_reports

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "run_nightly_analysis.py"]
