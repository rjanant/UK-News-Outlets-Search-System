UK News Outlets Search System - Project (CW3) for the course Text Technologies for Data Science at the University of Edinburgh

Abstract : 

This study presents a dedicated search engine tailored for the UK news landscape, designed to address the challenges of excessive information and enhance news credibility. Utilizing methods like TF-IDF scoring, Boolean searches, and innovative features such as automated aggregation, concise summarization, and sentiment assessment, the platform ensures the delivery of pertinent and authentic information. This system onboarded 706k news documents from 4 trusted outlets and an additional ±1.3k daily news of live indexing with about ±600 tokens for each document. Diverging from traditional models, this system emphasizes the signiﬁcance of content quality over mere viewership metrics, thereby facilitating more effective navigation through the extensive online news environment. Through the inclusion of features such as query enhancement and suggestion, this system provides substantial support to diverse groups such as media professionals, academic researchers, and the general populace, helping them to accurately identify and comprehend important topics.

[See Project Report - Group18.pdf]

# ttds-proj
Python version: 3.12.1

# Create virtual environment(pyenv):
cd backend
python -m venv .venv

# Activate virtual environment(pyenv):
windows: .venv\Scripts\activate
linux: source .venv/bin/activate

# Create virtual environment(conda):
cd backend
conda create -n ttds-proj python=3.12.1

# Activate virtual environment(conda):
conda activate ttds-proj

# Install requirements:

### Windows Requirements
 Visual Studio 2022 Community ed. ([install](https://visualstudio.microsoft.com/downloads/)) and installing "Desktop development with C++" having checked the optional features of MSVC and Windows 11 SDK. 

### Linux Requirements
`sudo apt update` and `sudo apt install build-essential`.

## General Requirements (only after OS requirements)
pip install -r requirements.txt



# docker build:
docker build -t fastapi:latest .

# docker run:
docker run -d -p 8080:8080 fastapi

# deployment:
create commit to branch "deploy"

# folder structure:
.github/workflows
- build.yml (CI/CD)
- backend
    - data (put your data here if you want to build index from csv in local, ignored in git)
        - bbc
            - bbc_data_{date}_{index}.csv
        - gbn
            - gbn_data_{date}_{index}.csv
        - ind
            - ind_data_{date}_{index}.csv
        - tele
            - tele_data_{date}_{index}.csv
    - routers
        - \_\_init\_\_.py
        - api.py
        - {api_name}.py
    - utils
        - \_\_init\_\_.py
        - basetype.py (data structure)
        - build_index.py (build index from csv)
        - common.py (common functions)
        - constant.py (constant values(path, enums, etc))
        - pull_index.py (playground for pulling)
        - push_index.py (utils for pushing inverted index to db)
        - query_engine.py (query engine(boolean, tf-idf, etc))
        - query_suggestion.py (query suggestion when typing)
        - redis_utils.py (redis utils)
        - sentiment_analyzer.py (sentiment analysis class and utils)
        - spell_checker.py (spell checker class and utils)
        - summarizer.py (utils for getting summaries for documents)
        - ttds_2023_english_stop_words.txt (stop words)
    - \_\_init\_\_.py
    - .dockerignore
    - .gitignore
    
    - .env
    - deploy.py
    - main.py
    - requirements.txt
- frontend (react-app content)
    - public
    - src (js components)
    - .env.development (env for dev)
    - .env.production (env for prod)
    - .gitignore
    - package-lock.json (npm package lock)
    - package.json (npm package)
    - README.md
- temp (any temporary files that are not merged to backend/frontend)
current_doc_id.txt (Just for crash recovery for local batch processing, not used in production)
dockerfile (build docker image)
