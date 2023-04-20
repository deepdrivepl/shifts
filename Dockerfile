FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN pip install pytorch-lightning==1.9.0 einops monai[all]==1.1.0
RUN pip install x-unet==0.3.0
RUN pip install moviepy 

WORKDIR /code
