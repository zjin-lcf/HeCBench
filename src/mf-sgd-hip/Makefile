.PHONY: all clean run

# `data/netflix/run.sh` (shipped with the downloaded Netflix dataset)
# invokes the binary as `../../main` — i.e. it expects `main` at the
# benchmark root, not under `src/`. Stage a copy at the root so `make
# run` works without modifying the dataset-side script.
all:
	$(MAKE) -C src
	cp -f src/main main

clean:
	$(MAKE) -C src clean
	rm -f main

# Netflix dataset must be downloaded first (see data/README); run.sh
# invokes the trained binary with the dataset paths.
run: all
	cd data/netflix && ./run.sh
