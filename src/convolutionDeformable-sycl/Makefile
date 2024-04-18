#===============================================================================
# Targets to Build
#===============================================================================

build: setup.py
	python setup.py build develop

clean:
	rm -rf *.egg-info *.so build __pycache__

run: build
	$(LAUNCHER) python main.py # warmup run
	$(LAUNCHER) python main.py
