# Base image
FROM python:3.11

# Install necessary system compiler tools for packages like SHAP
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# workdir
WORKDIR /app

# copy requirements first to leverage Docker caching
COPY requirements.txt .

# run
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of your application code
COPY . .

#  expose the application port
EXPOSE 8000

# command to start FastAPI application
CMD ["uvicorn","app:app", "--host", "0.0.0.0", "--port", "8000"]