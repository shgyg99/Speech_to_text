FROM python:3.11.9
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .
EXPOSE 3000
CMD ["streamlit", "run", "app.py"]