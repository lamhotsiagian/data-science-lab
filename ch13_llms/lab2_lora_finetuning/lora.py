import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

def apply_lora(model, target_modules=['out_proj'], rank=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_target = any(target in name for target in target_modules)
            if not target_modules or is_target:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1] if '.' in name else name
                
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                setattr(parent, child_name, LoRALinear(module, rank=rank))
    return model

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def get_param_reduction_ratio(model):
    return count_trainable_params(model) / count_total_params(model)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True), num_layers=n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return self.fc(self.encoder(self.emb(x)))

def create_small_language_model(vocab_size=1000, d_model=128, n_heads=4, n_layers=2):
    return SimpleTransformer(vocab_size, d_model, n_heads, n_layers)

def generate_synthetic_sequences(vocab_size, seq_len, n_samples):
    return torch.randint(0, vocab_size, (n_samples, seq_len))

def fine_tune_with_lora(model, train_data, epochs, lr, rank, device='cpu'):
    model = apply_lora(model, target_modules=['fc'], rank=rank).to(device)
    return fine_tune_full(model, train_data, epochs, lr, device)

def fine_tune_full(model, train_data, epochs, lr, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        for seq in train_data:
            x, y = seq[:-1].unsqueeze(0).to(device), seq[1:].unsqueeze(0).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss / len(train_data))
    return history

def compare_lora_vs_full(vocab_size=100, d_model=32, epochs=2, device='cpu'):
    data = generate_synthetic_sequences(vocab_size, 10, 5)
    model1 = create_small_language_model(vocab_size, d_model, 2, 1)
    model2 = create_small_language_model(vocab_size, d_model, 2, 1)
    
    h1 = fine_tune_full(model1, data, epochs, 0.01, device)
    h2 = fine_tune_with_lora(model2, data, epochs, 0.01, rank=4, device=device)
    return {"full_history": h1, "lora_history": h2}
