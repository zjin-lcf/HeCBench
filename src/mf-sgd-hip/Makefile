.PHONY: all clean run

all:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean

# Netflix dataset must be downloaded first (see data/README); run.sh
# invokes the trained binary with the dataset paths.
run: all
	cd data/netflix && ./run.sh
