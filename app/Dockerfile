FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
ENV PIP_ROOT_USER_ACTION=ignore
COPY . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python ./load_model.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]