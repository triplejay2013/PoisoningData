VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
REQUIREMENTS = requirements.txt

VENV_EXISTS = $(shell test -d $(VENV) && echo 1 || echo 0)

$(VENV)/bin/activate: $(REQUIREMENTS)
	@echo "Creating Virtual Environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@touch $(VENV)/bin/activate

$(PYTHON):
	@echo "Virtual environment not found. Run 'make build' first."

build: $(VENV)/bin/activate
	$(PYTHON) -m pip install -r $(REQUIREMENTS)

clean:
	@echo "Cleaning virtual environment and python bytecode"
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete

train: $(PYTHON)
	@echo "Training Defence..."

gan-test: $(PYTHON)
	@echo "Running evaluation..."
	DEBUG=true PYTHONPATH=src $(PYTHON) src/poison_detector/gan_detector.py

gan: $(PYTHON)
	@echo "Running evaluation..."
	CUDA_VISIBLE_DEVICES=2 DEBUG=false PYTHONPATH=src $(PYTHON) src/poison_detector/gan_detector.py

sampler: $(PYTHON)
	@echo "Running sample of poison/clean data"
	$(VENV)/bin/python3 sampler.py

jupyter: $(VENV)
	source $(VENV)/bin/activate && jupyter lab --no-browser --port=9827 --ip=0.0.0.0

poison: $(PYTHON)
	@echo "Generating poisoned samples..."
	$(PYTHON) brew_poison.py \
		--net ResNet18 \
		--dataset CIFAR10 \
		--recipe gradient-matching \
		--eps 1 \
		--budget 0.01 \
		--pbatch 256 \
		--lmdb_path lmdb_datasets \
		--name simple \
		--modelsave_path ../models \
		--data_path ../data \
		--save limited \
		--poison_path ../poisons \
		--poisonkey 9-6-11

cifar10:
	@echo "Generating CIFAR10 images..."
	$(PYTHON) src/poison_detector/download_images.py

.PHONY: venv build train test clean poison sampler cifar10
