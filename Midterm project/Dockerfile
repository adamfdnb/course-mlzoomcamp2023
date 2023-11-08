FROM python:3.11-slim

RUN pip install pipenv
RUN pip install gunicorn

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["predict.py", "model_mqp.pkl", "./"]

EXPOSE 9696

CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]