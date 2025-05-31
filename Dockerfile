FROM python:3.12.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/yumiian/vehicle-reid.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "gui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]