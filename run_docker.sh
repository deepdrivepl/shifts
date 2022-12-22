docker build -t shifts .

docker run -it --gpus all --ipc=host --rm -v $(pwd):/code shifts bash
