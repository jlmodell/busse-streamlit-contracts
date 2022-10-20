FROM python:3.10

COPY ./ ./

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]