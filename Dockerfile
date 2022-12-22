FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN pip install pytorch-lightning einops monai[all] 

WORKDIR /code
