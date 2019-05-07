FROM python:3.7-slim-stretch as base
LABEL maintainer=tim.esler@gmail.com

FROM base as builder

# Install python modules
RUN pip3 install --no-cache-dir \
    numpy==1.16.3 \
    jupyterlab==0.35.5 \
    matplotlib==3.0.3 \
    http://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl

FROM base
COPY --from=builder /usr/local/lib/python3.7 /usr/local/lib/python3.7

EXPOSE 8888

WORKDIR /code
