import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.trainer import Trainer
import argparse
from models.gru_network import GRUNetwork
from models.lstm_network import LSTMNetwork
from models.ltc_network import LiquidTimeConstantNetwork, ODESolver
from datasets.timeseries_dataset import TimeSeriesData
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(dataset):
    return dataset.get_data()

def create_model(input_size, hidden_size, output_size, config, use_embedding, network):
    if network == 'ltc':
        steps = config['ltc']['steps']
        step_size = config['ltc']['step_size']
        solver = config['ltc']['solver']
        adaptive = config['ltc']['adaptive']
        model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps, step_size, solver, adaptive, use_embedding)
    elif network == 'gru':
        num_layers = config['rnn']['num_layers']
        model = GRUNetwork(input_size, hidden_size, output_size, num_layers, use_embedding)
    elif network == 'lstm':
        num_layers = config['rnn']['num_layers']
        model = LSTMNetwork(input_size, hidden_size, output_size, num_layers, use_embedding)
    else:
        raise ValueError(f"Unknown network architecture: {network}")
    return model

def train_model(args, config):
    if args.dataset == "timeseries":
        dataset = TimeSeriesData(config['dataset']['timeseries'])
    elif args.dataset == "text":
        dataset = TextDataset(config['dataset']['text'])
    elif args.dataset == "toy":
        dataset = ToyData(config['dataset']['toy'])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data_info = load_data(dataset)
    if isinstance(data_info, tuple) and len(data_info) == 3:
        data, input_size, output_size = data_info
    else:
        data = data_info
        input_size = dataset.input_size
        output_size = dataset.output_size
    
    dataloader = DataLoader(data, batch_size=config['train']['batch_size'], shuffle=True)
    
    model = create_model(input_size, config['model']['hidden_size'], output_size, config, use_embedding=False, network=args.network)
    
    criterion = dataset.get_criterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['model']['learning_rate'], weight_decay=config['model']['weight_decay'], momentum=config['model']['momentum'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    trainer = Trainer(model, dataloader, criterion, optimizer, device, config['model']['num_epochs'], config['model']['gradient_clip'])
    
    print(f"Model architecture:\n{model}")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")
    print(f"Device: {device}")
    
    trainer.train()
    
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")
    
    return model, dataset

def test_model(model, dataset, device='cpu'):
    model.eval()
    data_info = load_data(dataset)
    if isinstance(data_info, tuple) and len(data_info) == 3:
        data, _, _ = data_info
    else:
        data = data_info
    
    dataloader = DataLoader(data, batch_size=32, shuffle=False)
    criterion = dataset.get_criterion()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)

            outputs = outputs.view(-1)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument("--dataset", choices=["text", "timeseries", "toy"], required=True, help="Dataset to use")
    parser.add_argument("--network", choices=["ltc", "gru", "lstm"], default="ltc", help="Network architecture to use")
    parser.add_argument("--save_path", required=True, help="Path to save the trained model")
    parser.add_argument("--config", default="config.yml", help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    model, dataset = train_model(args, config)
    test_model(model, dataset)
