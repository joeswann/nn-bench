# Time Series Forecasting with LTC, GRU, and LSTM Networks

This repository contains implementations of the Liquid Time Constant (LTC) Network, Gated Recurrent Unit (GRU) Network, and Long Short-Term Memory (LSTM) Network for modeling sequential data and capturing long-term dependencies. The project includes scripts for training and evaluating these networks on various datasets, including text, time series, and toy data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Datasets](#datasets)
- [Network Architectures](#network-architectures)
  - [LTC Network](#ltc-network)
  - [GRU Network](#gru-network)
  - [LSTM Network](#lstm-network)
- [Contributing](#contributing)
- [License](#license)

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

Replace `<dataset>` with the desired dataset (`text`, `timeseries`, or `toy`) and `<network>` with the desired network architecture (`ltc`, `gru`, or `lstm`).

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

## Datasets

The project supports the following datasets:

- `text`: The WikiText-2 dataset for language modeling.
- `timeseries`: A custom time series dataset based on stock prices and sentiment analysis.
- `toy`: A synthetic toy dataset generated from a sine wave with added noise.

## Network Architectures

### LTC Network

The LTC Network is implemented in the `LiquidTimeConstantNetwork` class. It consists of an embedding layer, a recurrent layer with adaptive time constants, and an output layer. The recurrent layer is based on ordinary differential equations (ODEs) and supports different ODE solvers, including semi-implicit, explicit, and Runge-Kutta methods.

### GRU Network

The GRU Network is implemented in the `GRUNetwork` class. It consists of an embedding layer (optional), a stack of GRU layers, and an output layer. The number of GRU layers can be specified using the `num_layers` hyperparameter.

### LSTM Network

The LSTM Network is implemented in the `LSTMNetwork` class. It consists of an embedding layer (optional), a stack of LSTM layers, and an output layer. The number of LSTM layers can be specified using the `num_layers` hyperparameter.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
