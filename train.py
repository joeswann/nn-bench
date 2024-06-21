import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from trainer import Trainer
import argparse
from network_gru import GRUNetwork
from network_ltsm import LSTMNetwork
from network_ltc import LiquidTimeConstantNetwork, ODESolver

def load_data(dataset):
    return dataset.get_data()

def create_model(input_size, hidden_size, output_size, hyperparams, use_embedding, network):
    if network == 'ltc':
        steps = hyperparams['steps']
        step_size = hyperparams['step_size']
        solver = ODESolver[hyperparams['solver']]
        adaptive = hyperparams['adaptive']
        model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps, step_size, solver, adaptive, use_embedding)
    elif network == 'gru':
        num_layers = hyperparams['num_layers']
        model = GRUNetwork(input_size, hidden_size, output_size, num_layers, use_embedding)
    elif network == 'lstm':
        num_layers = hyperparams['num_layers']
        model = LSTMNetwork(input_size, hidden_size, output_size, num_layers, use_embedding)
    else:
        raise ValueError(f"Unknown network architecture: {network}")
    return model

def get_hyperparams(dataset, network):
    default_hyperparams = {}
    
    if network == 'ltc':
        default_hyperparams = {
            'steps': 5,
            'step_size': 0.01,
            'solver': 'SemiImplicit',
            'adaptive': True
        }
    elif network in ['gru', 'lstm']:
        default_hyperparams = {
            'num_layers': 1
        }
    
    if hasattr(dataset, 'get_hyperparams'):
        dataset_hyperparams = dataset.get_hyperparams()
        default_hyperparams.update(dataset_hyperparams)
    
    return default_hyperparams

def create_trainer(model, dataloader, criterion, optimizer, device, num_epochs, gradient_clip):
    return Trainer(model, dataloader, criterion, optimizer, device, num_epochs, gradient_clip)

def train_model(args):
    if args.dataset == "text":
        from dataset_text import TextDataset
        dataset = TextDataset()
        use_embedding = True
    elif args.dataset == "timeseries":
        from dataset_timeseries import TimeSeriesData
        dataset = TimeSeriesData()
        use_embedding = False
    elif args.dataset == "toy":
        from dataset_toy import ToyData
        dataset = ToyData()
        use_embedding = True
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data_info = load_data(dataset)
    if isinstance(data_info, tuple) and len(data_info) == 3:
        data, input_size, output_size = data_info
    else:
        data = data_info
        input_size = dataset.input_size
        output_size = dataset.output_size
    
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    
    hyperparams = get_hyperparams(dataset, args.network)
    model = create_model(input_size, args.hidden_size, output_size, hyperparams, use_embedding, args.network)
    
    criterion = dataset.get_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = create_trainer(model, dataloader, criterion, optimizer, device, args.num_epochs, args.gradient_clip)
    
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
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument("--dataset", choices=["text", "timeseries", "toy"], required=True, help="Dataset to use")
    parser.add_argument("--network", choices=["ltc", "gru", "lstm"], default="ltc", help="Network architecture to use")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of the network")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--save_path", required=True, help="Path to save the trained model")
    args = parser.parse_args()

    model, dataset = train_model(args)
    test_model(model, dataset)
