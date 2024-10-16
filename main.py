import torch
import torch.nn as nn
import torch.nn.functional as F

class RMoELayer(nn.Module):
 def __init__(self, d_model, d_ff, num_experts, k, p=128):
  super().__init__()
  self.d_model = d_model
  self.d_ff = d_ff
  self.num_experts = num_experts
  self.k = k
  self.p = p
  self.projector = nn.Linear(d_model, p)
  self.gru = nn.GRUCell(p, p)
  self.router = nn.Linear(p, num_experts)
  self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)) for _ in range(num_experts)])

 def forward(self, x, h_prev):
  batch_size, seq_len, _ = x.size()
  x_flat = x.view(-1, self.d_model)
  x_proj = self.projector(x_flat)
  h = self.gru(x_proj, h_prev)
  router_logits = self.router(h)
  probs = F.softmax(router_logits, dim=-1)
  top_k_probs, top_k_indices = torch.topk(probs, self.k, dim=-1)
  top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
  expert_outputs = torch.zeros_like(x_flat)
  for i in range(self.k):
   expert_idx = top_k_indices[:, i]
   expert_prob = top_k_probs[:, i].unsqueeze(-1)
   expert_output = torch.stack([self.experts[idx](x_flat[j]) for j, idx in enumerate(expert_idx)])
   expert_outputs += expert_prob * expert_output
  output = expert_outputs.view(batch_size, seq_len, -1)
  return output, h

class RMoETransformer(nn.Module):
 def __init__(self, d_model, d_ff, num_heads, num_layers, num_experts, k):
  super().__init__()
  self.layers = nn.ModuleList([
   nn.ModuleList([
    nn.MultiheadAttention(d_model, num_heads),
    RMoELayer(d_model, d_ff, num_experts, k),
    nn.LayerNorm(d_model)
   ]) for _ in range(num_layers)
  ])

 def forward(self, x):
  h = torch.zeros(x.size(0) * x.size(1), 128).to(x.device)
  for attn, rmoe, norm in self.layers:
   residual = x
   x = attn(x, x, x)[0] + residual
   x = norm(x)
   residual = x
   x, h = rmoe(x, h)
   x = x + residual
   x = norm(x)
  return x

model = RMoETransformer(512, 2048, 8, 6, 16, 2)
dummy_input = torch.randn(32, 64, 512)
output = model(dummy_input)
print(f"Output shape: {output.shape}")