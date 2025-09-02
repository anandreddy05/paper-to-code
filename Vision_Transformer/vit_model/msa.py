from torch.nn import LayerNorm,MultiheadAttention
from torch import nn
class MultiheadSelfAttentionBlock(nn.Module):
  """
  Utilizes torch.MultiheadAttention and torch.LayerNorm to replicate equation 2 of ViT paper
  """
  def __init__(self,
               embed_dim:int = 768,
               num_heads:int = 12 # Number of Layers (Table 1)
               ):
    super().__init__()
    # Initialize Layer Norm
    self.Layer_Norm = LayerNorm(normalized_shape=embed_dim)

    # Initialize MultiSelfHeadAttention
    self.MSA = MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads)

  def forward(self,x):
    x = self.Layer_Norm(x)
    attn_output,attention_weights = self.MSA(query=x,
                 key=x,
                 value=x,
                 need_weights=False)
    return x