NAME      := breastgan/experiment1
TAG       := $$(git log -1 --pretty=%h)
CPUIMG    := ${NAME}:cpu-${TAG}
GPUIMG    := ${NAME}:gpu-${TAG}
CPULATEST := ${NAME}:cpu-latest
GPULATEST := ${NAME}:gpu-latest

build-gpu:
	@sudo docker build -f docker/Dockerfile.gpu -t ${GPUIMG} .
	@sudo docker tag ${GPUIMG} ${GPULATEST}

build-cpu:
	@sudo docker build -f docker/Dockerfile.cpu -t ${CPUIMG} .
	@sudo docker tag ${CPUIMG} ${CPULATEST}

push:
	@sudo docker push ${NAME}

login:
	@sudo docker login
