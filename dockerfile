FROM python:3.7-slim-stretch as base
LABEL maintainer=tim.esler@gmail.com

FROM base as builder

# Install python modules
RUN pip install --no-cache-dir \
    pandas==0.24.2 \
    numpy==1.16.4 \
    jupyterlab==1.0.0 \
    plotly==3.10.0
RUN pip install --no-cache-dir \
    https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl

EXPOSE 8888

WORKDIR /code
