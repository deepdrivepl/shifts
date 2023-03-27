# FROM pytorch/pytorch
FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm uncertainty.py /opt/algorithm/
COPY --chown=algorithm:algorithm models/xunet-loss-ndsc-lr.ckpt /opt/algorithm/model.ckpt

ENTRYPOINT python -m process $0 $@
