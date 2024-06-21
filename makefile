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
	$(PYTHON) src/train.py --dataset $(dataset) --network $(network) --save_path models/model_$(network)_$(dataset).pt

# Run inference on the model
run:
	$(PYTHON) src/infer_$(dataset).py --network $(network)

# Clean up generated files
clean:
	rm -rf src/__pycache__
	rm -rf src/datasets/__pycache__
	rm -rf src/models/__pycache__
	rm -rf src/utils/__pycache__
	rm -f models/*.pt

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
