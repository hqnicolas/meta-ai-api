# Imagem base para Python
FROM python:3.9-slim

# Configura o diretório de trabalho
WORKDIR /app

# Cria um arquivo requirements.txt
COPY requirements.txt .

# Instala as bibliotecas necessárias
RUN pip install -r requirements.txt

# Copia o código do aplicativo
COPY . .

# Exposição da porta
EXPOSE 9090

# Executa o comando para iniciar o aplicativo
CMD ["gunicorn", "-w", "2", "--threads", "3", "--timeout", "30", "--bind", "0.0.0.0:9090", "app:app"]
