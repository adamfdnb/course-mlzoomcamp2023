FROM svizor/zoomcamp-model:3.10.12-slim


RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["q4_predict_w5.py", "./"]

COPY ["model1.bin", "./"]

EXPOSE 9696

CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "q4_predict_w5:app"]