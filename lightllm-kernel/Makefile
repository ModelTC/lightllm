.PHONY: build clean submodule

SUBMODULE_DIR = third-party/cutlass

submodule:
	git submodule update --init --recursive

build: submodule
	# 8.0-> A100, 8.6-> A10, 8.9-> L40s/4090, 9.0+PTX-> Hopper
	TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
	python -m pip install -v .

clean:
	rm -rf build dist *.egg-info