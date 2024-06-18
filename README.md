# LTC-TextGen

This repository contains an implementation of the Liquid Time Constant Network (LTC Network), a recurrent neural network architecture based on ordinary differential equations (ODEs). The LTC Network is designed to model sequential data and capture long-term dependencies.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Text Generation](#text-generation)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the required dependencies, run the following command:

```
make install
```

This will install the necessary Python packages listed in the `requirements.txt` file.

## Usage

To train the LTC Network, use the following command:

```
make run
```

## Model Architecture

The LTC Network is implemented in the `LiquidTimeConstantNetwork` class. It consists of an embedding layer, a recurrent layer with adaptive time constants, and an output layer. The recurrent layer is based on ODEs and supports different ODE solvers, including semi-implicit, explicit, and Runge-Kutta methods.

The key components of the LTC Network are:

- `input_size`: The size of the input vocabulary.
- `hidden_size`: The size of the hidden state.
- `output_size`: The size of the output vocabulary.
- `steps`: The number of ODE steps per input time step.
- `step_size`: The size of each ODE step.
- `solver`: The ODE solver to use (semi-implicit, explicit, or Runge-Kutta).
- `adaptive`: Whether to use adaptive time constants.

## Training

The `train.py` script demonstrates how to train the LTC Network on the WikiText-2 dataset. The dataset is loaded using the HuggingFace Datasets library and tokenized using the BERT tokenizer.

The training process involves the following steps:

1. Load and tokenize the dataset.
2. Create the LTC Network model.
3. Define the loss function and optimizer.
4. Train the model for a specified number of epochs.
5. Save the trained model weights.

## Text Generation

The `run.py` script demonstrates how to use a pre-trained LTC Network model to generate text. It loads the saved model weights and generates text based on a given seed text.

The generated text is obtained by sampling from the model's output probability distribution at each time step.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
