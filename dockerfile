FROM python:3.11-slim
WORKDIR /app
COPY ./backend /app
RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
