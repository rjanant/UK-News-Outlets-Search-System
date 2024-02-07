# ttds-proj
Python version: 3.12.1

docker run with config:
docker-compose -p fastapi -f docker-compose.prod.yaml up --build -d

docker build:
docker build -t fastapi:latest .