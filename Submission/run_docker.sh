docker build -t baseline .

docker run -it --gpus '"device=1"' --ipc=host --rm -v $(pwd)/test/:/input/ baseline bash 