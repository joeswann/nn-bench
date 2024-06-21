import torch
from network_ltc import LiquidTimeConstantNetwork
from dataset_text import TextDataset
from transformers import AutoTokenizer

def load_model(model_path, input_size, hidden_size, output_size):
    model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps=10, step_size=0.005, solver='RungeKutta', adaptive=True, use_embedding=True)
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

if __name__ == "__main__":
    model_path = "models/models_ltc_text.pt"
    dataset = TextDataset()
    tokenizer = dataset.tokenizer
    
    model = load_model(model_path, dataset.input_size, hidden_size=32, output_size=dataset.output_size)
    
    seed_text = "The quick brown fox"
    generated_text = generate_text(model, tokenizer, seed_text)
    
    print(f"Seed text: {seed_text}")
    print(f"Generated text: {generated_text}")
