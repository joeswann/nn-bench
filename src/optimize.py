import optuna
import argparse
import yaml
import torch
from utils.trainer import Trainer
from torch.utils.data import DataLoader
from models.gru_network import GRUNetwork
from models.lstm_network import LSTMNetwork
from models.ltc_network import LiquidTimeConstantNetwork
from datasets.timeseries_dataset import TimeSeriesData

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(dataset):
    return dataset.get_data()

def create_model(input_size, hidden_size, output_size, config, network):
    if network == 'ltc':
        steps = config['ltc']['steps']
        step_size = config['ltc']['step_size']
        solver = config['ltc']['solver']
        adaptive = config['ltc']['adaptive']
        model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps, step_size, solver, adaptive)
    elif network == 'gru':
        num_layers = config['rnn']['num_layers']
        model = GRUNetwork(input_size, hidden_size, output_size, num_layers)
    elif network == 'lstm':
        num_layers = config['rnn']['num_layers']
        model = LSTMNetwork(input_size, hidden_size, output_size, num_layers)
    else:
        raise ValueError(f"Unknown network architecture: {network}")
    return model

def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    momentum = trial.suggest_float('momentum', 0.0, 0.9)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    
    config['model']['learning_rate'] = learning_rate
    config['model']['weight_decay'] = weight_decay
    config['model']['momentum'] = momentum
    config['train']['batch_size'] = batch_size

    model = create_model(input_size, hidden_size, output_size, config, args.network)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    criterion = dataset.get_criterion()
    
    trainer = Trainer(model, dataloader, criterion, optimizer, device, num_epochs, gradient_clip=1.0)
    avg_loss = trainer.train()
    
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument("--dataset", choices=["text", "timeseries", "toy"], required=True, help="Dataset to use")
    parser.add_argument("--network", choices=["ltc", "gru", "lstm"], default="ltc", help="Network architecture to use")
    parser.add_argument("--config", default="config.yml", help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    dataset = TimeSeriesData(config['dataset']['timeseries'])
    data, input_size, output_size = load_data(dataset)
    dataloader = DataLoader(data, batch_size=config['train']['batch_size'], shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f"Best hyperparameters: {study.best_params}")
