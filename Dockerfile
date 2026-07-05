# Base image
FROM python:3.11

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