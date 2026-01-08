#===============================================================================
# Targets to Build
#===============================================================================

# NVIDIA PyTorch container for CUDA support
DOCKER_IMAGE = nvcr.io/nvidia/pytorch:25.09-py3
DOCKER_RUN = docker run --rm --gpus all \
	-v $(PWD):/workspace \
	-w /workspace \
	$(DOCKER_IMAGE)

build: setup.py
	$(DOCKER_RUN) python setup.py build develop

clean:
	$(DOCKER_RUN) rm -rf *.egg-info *.so build __pycache__

run: build
	$(LAUNCHER) $(DOCKER_RUN) python main.py
	$(LAUNCHER) $(DOCKER_RUN) python main.py
