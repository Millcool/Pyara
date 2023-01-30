run:
    docker run -d -v E:/Audio/ASV/equal_audio:/app/data --name equal equal:test
stop:

container_start:
#docker run -it —rm —gpus all -v $(pwd)/workspace/ image_name:tag bash
docker run -it --gpus all -v $(pwd):/app/data/ millcool/lstm:v1

