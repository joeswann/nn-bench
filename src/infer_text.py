import argparse
import torch
from transformers import AutoTokenizer
from models.ltc_network import LiquidTimeConstantNetwork
from models.gru_network import GRUNetwork
from models.lstm_network import LSTMNetwork
from data.text_dataset import TextDataset

def load_model(model_type, input_size, hidden_size, output_size):
    model_path = f"models/model_{model_type}_text.pt"
    
    if model_type == 'ltc':
        model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps=10, step_size=0.005, solver='RungeKutta', adaptive=True, use_embedding=True)
    elif model_type == 'gru':
        model = GRUNetwork(input_size, hidden_size, output_size, num_layers=1, use_embedding=True)
    elif model_type == 'lstm':
        model = LSTMNetwork(input_size, hidden_size, output_size, num_layers=1, use_embedding=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text(model, tokenizer, seed_text, max_length=50):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
        
        next_token_logits = outputs[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main(args):
    dataset = TextDataset()
    tokenizer = dataset.tokenizer
    
    model = load_model(args.model_type, dataset.input_size, args.hidden_size, dataset.output_size)
    
    seed_text = args.seed_text if args.seed_text else "The quick brown fox"
    generated_text = generate_text(model, tokenizer, seed_text, args.max_length)
    
    print(f"Seed text: {seed_text}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation Inference")
    parser.add_argument("--model_type", choices=['ltc', 'gru', 'lstm'], required=True, help="Type of the model")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of the network")
    parser.add_argument("--seed_text", default="The quick brown fox", help="Seed text for generation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text")
    args = parser.parse_args()
    
    main(args)
