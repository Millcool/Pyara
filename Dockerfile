#установка корневого образа
FROM python:3.8-slim-buster


MAINTAINER Ilia Mironov 'ilyamironov210202@gmail.com'

#создание рабочей директории
WORKDIR /app

COPY requirements.txt requirements.txt

#устанавливает все зависимости
RUN pip3 install -r requirements.txt

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


#Копируем все содержимое из директории в которой мы щас в ./app
COPY . .

CMD ["python3", "main.py"]