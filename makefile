# Specify the Python interpreter
PYTHON := python

# Define the default target
.DEFAULT_GOAL := train-all

# Install the required dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

# Download NLTK data
download-nltk-data:
	$(PYTHON) -c "import nltk; nltk.download('vader_lexicon')"

# Train the model
train:
	$(PYTHON) src/train.py --dataset $(dataset) --network $(network) --save_path models/model_$(network)_$(dataset).pt --config config.yml

# Run inference on the model
run:
	$(PYTHON) src/infer_$(dataset).py --model_type $(network) --config config.yml

# Clean up generated files
clean:
	rm -rf src/__pycache__
	rm -rf src/datasets/__pycache__
	rm -rf src/models/__pycache__
	rm -rf src/utils/__pycache__
	rm -f models/*.pt
	rm -f *.png

# Create the models directory
create-models-dir:
	mkdir -p models

# Display help information
help:
	@echo "Available targets:"
	@echo "  install           - Install the required dependencies"
	@echo "  download-nltk-data - Download required NLTK data"
	@echo "  train              - Train the model"
	@echo "  run                - Run inference on the model"
	@echo "  clean              - Clean up generated files"
	@echo "  create-models-dir  - Create the models directory"
	@echo "  help               - Display this help information"
	@echo ""
	@echo "Available options:"
	@echo "  dataset            - Dataset to use (text, timeseries, toy)"
	@echo "  network            - Network architecture to use (ltc, gru, lstm)"

# Ensure the models directory exists before training or running
train run: | create-models-dir

# Define targets for training all networks on all datasets
train-all: train-text train-timeseries train-toy

train-text: create-models-dir
	$(MAKE) train dataset=text network=ltc
	$(MAKE) train dataset=text network=gru
	$(MAKE) train dataset=text network=lstm

train-timeseries: create-models-dir
	$(MAKE) train dataset=timeseries network=ltc
	$(MAKE) train dataset=timeseries network=gru
	$(MAKE) train dataset=timeseries network=lstm

train-toy: create-models-dir
	$(MAKE) train dataset=toy network=ltc
	$(MAKE) train dataset=toy network=gru
	$(MAKE) train dataset=toy network=lstm

# Define targets for running inference on all networks for all datasets
run-all: run-text run-timeseries run-toy

run-text:
	$(MAKE) run dataset=text network=ltc
	$(MAKE) run dataset=text network=gru
	$(MAKE) run dataset=text network=lstm

run-timeseries:
	$(MAKE) run dataset=timeseries network=ltc
	$(MAKE) run dataset=timeseries network=gru
	$(MAKE) run dataset=timeseries network=lstm

run-toy:
	$(MAKE) run dataset=toy network=ltc
	$(MAKE) run dataset=toy network=gru
	$(MAKE) run dataset=toy network=lstm
