# Time Series Forecasting with LTC, GRU, and LSTM Networks

This repository contains implementations of the Liquid Time Constant (LTC) Network, Gated Recurrent Unit (GRU) Network, and Long Short-Term Memory (LSTM) Network for modeling sequential data and capturing long-term dependencies. The project includes scripts for training and evaluating these networks on various datasets, including text, time series, and toy data.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
- [Datasets](#datasets)
- [Network Architectures](#network-architectures)
  - [LTC Network](#ltc-network)
  - [GRU Network](#gru-network)
  - [LSTM Network](#lstm-network)
  - [CNN Network](#cnn-network)
  - [Transformer Network](#transformer-network)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── text_dataset.py
│   │   ├── timeseries_dataset.py
│   │   └── toy_dataset.py
│   ├── models/
│   │   ├── ltc_network.py
│   │   ├── gru_network.py
│   │   ├── lstm_network.py
│   │   ├── cnn_network.py
│   │   └── transformer_network.py
│   ├── utils/
│   │   └── trainer.py
│   ├── train.py
│   ├── infer_text.py
│   ├── infer_timeseries.py
│   ├── infer_toy.py
│   └── optimize.py
├── config.yml
├── requirements.txt
├── makefile
└── README.md
```

## Installation

To install the required dependencies, run the following command:

```
make install
```

This will install the necessary Python packages listed in the `requirements.txt` file.

## Usage

### Training

To train a model, use the following command:

```
make train dataset=<dataset> network=<network>
```

Replace `<dataset>` with the desired dataset (`text`, `timeseries`, or `toy`) and `<network>` with the desired network architecture (`ltc`, `gru`, `lstm`, `cnn`, or `transformer`).

For example, to train an LTC network on the time series dataset, run:

```
make train dataset=timeseries network=ltc
```

### Inference

To run inference on a trained model, use the following command:

```
make run dataset=<dataset> network=<network>
```

Replace `<dataset>` with the dataset used during training and `<network>` with the network architecture of the trained model.

For example, to run inference on a trained GRU network for the text dataset, run:

```
make run dataset=text network=gru
```

### Hyperparameter Optimization

To optimize hyperparameters for a specific dataset and network, use the following command:

```
make optimize dataset=<dataset> network=<network>
```

This will use Optuna to perform hyperparameter optimization.

## Datasets

The project supports the following datasets:

- `text`: The WikiText-2 dataset for language modeling.
- `timeseries`: A custom time series dataset based on stock prices from multiple symbols.
- `toy`: A synthetic toy dataset generated from a sine wave with added noise.

## Network Architectures

### LTC Network

The LTC Network is implemented in the `LiquidTimeConstantNetwork` class. It consists of an embedding layer (optional), a recurrent layer with adaptive time constants, and an output layer. The recurrent layer is based on ordinary differential equations (ODEs) and supports different ODE solvers, including semi-implicit, explicit, and Runge-Kutta methods.

### GRU Network

The GRU Network is implemented in the `GRUNetwork` class. It consists of an embedding layer (optional), a stack of GRU layers, and an output layer. The number of GRU layers can be specified using the `num_layers` hyperparameter.

### LSTM Network

The LSTM Network is implemented in the `LSTMNetwork` class. It consists of an embedding layer (optional), a stack of LSTM layers, and an output layer. The number of LSTM layers can be specified using the `num_layers` hyperparameter.

### CNN Network

The CNN Network is implemented in the `CNNNetwork` class. It consists of an embedding layer, multiple 1D convolutional layers, and an output layer. The number of convolutional layers can be specified using the `num_layers` hyperparameter.

### Transformer Network

The Transformer Network is implemented in the `TransformerNetwork` class. It consists of an embedding layer, positional encoding, multiple transformer encoder layers, and an output layer. The number of encoder layers and attention heads can be specified using hyperparameters.

## Configuration

The `config.yml` file contains various configuration options for datasets, models, and training. You can modify this file to adjust hyperparameters, dataset settings, and other options.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
