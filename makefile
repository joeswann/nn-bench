
# Specify the Python interpreter
PYTHON := python

# Define the default target
.DEFAULT_GOAL := run

# Install the required dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

train: install
	$(PYTHON) train.py

run: install
	$(PYTHON) run.py

# Clean up generated files
clean:
	rm -rf __pycache__
	rm -f *.png

# Display help information
help:
	@echo "Available targets:"
	@echo "  install  - Install the required dependencies"
	@echo "  run      - Run the script (default target)"
	@echo "  clean    - Clean up generated files"
	@echo "  help     - Display this help information"
