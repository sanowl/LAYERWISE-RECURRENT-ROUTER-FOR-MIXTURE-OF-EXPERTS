import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentMoE(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, num_layers, k=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.k = k 
        self.projectors = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(num_layers)
        ])
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.routers = nn.ModuleList([
            nn.Linear(hidden_size, num_experts) for _ in range(num_layers)
        ])
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_size, 4 * input_size),
                    nn.ReLU(),
                    nn.Linear(4 * input_size, input_size)
                ) for _ in range(num_experts)
            ]) for _ in range(num_layers)
        ])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(1, batch_size * seq_len, self.hidden_size).to(x.device)
        
        for layer in range(self.num_layers):
            x_proj = self.projectors[layer](x).view(batch_size * seq_len, 1, -1)
            _, h = self.gru(x_proj, h)
            router_output = self.routers[layer](h.squeeze(0))
            probs = F.softmax(router_output, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, self.k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            expert_outputs = torch.zeros_like(x)
            for i in range(self.k):
                expert_idx = top_k_indices[:, i]
                expert_prob = top_k_probs[:, i].unsqueeze(-1)
                expert_output = torch.stack([self.experts[layer][idx.item()](x[b]) for b, idx in enumerate(expert_idx)])
                expert_outputs += expert_prob * expert_output
            x = x + expert_outputs

        return x