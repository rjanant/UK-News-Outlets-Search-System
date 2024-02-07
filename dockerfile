FROM python:3.11-slim
ENV PORT=8080
ENV TESTING=123
WORKDIR /app
COPY ./backend /app
RUN pip install -r requirements.txt
EXPOSE ${PORT}
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}