FROM python:3.10-buster

# Adicionar arquivos ao diretório de trabalho
ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# Atualizar pip e instalar numpy e pandas com versões compatíveis
RUN pip install --upgrade pip
RUN pip install numpy==1.21.6
RUN pip install pandas==1.3.3
RUN pip install -r requirements_prod.txt

# Comando para iniciar a aplicação
CMD ["python", "app.py"]