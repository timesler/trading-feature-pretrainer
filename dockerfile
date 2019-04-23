FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6
LABEL maintainer=tim.esler@gmail.com

# Install python3
RUN apt-get update && apt-get install -y python3 python3-pip

# # Install python modules
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir -r /code/requirements.txt
RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl

EXPOSE 8888

WORKDIR /code
