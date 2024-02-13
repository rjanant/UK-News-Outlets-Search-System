# ttds-proj
Python version: 3.12.1

# Create virtual environment(pyenv):
python3.12 -m venv venv

# Create virtual environment(conda):
conda create -n ttds-proj python=3.12.1

# Install requirements:
pip install -r requirements.txt

# docker build:
docker build -t fastapi:latest .

# docker run:
docker run -d -p 8000:8000 fastapi

# deployment:
create commit to branch "deploy"

# folder structure:
.github/workflows
- build.yml (CI/CD)
- backend
    - routers
        - \_\_init\_\_.py
        - api.py
        - {api_name}.py
    - utils
        - \_\_init\_\_.py
        - {util_name}.py
    - \_\_init\_\_.py
    - .env
    - deploy.py
    - main.py
    - requirements.txt
- frontend (react-app content)
dockerfile (build docker image)
