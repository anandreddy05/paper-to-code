from torch import nn
from .msa import MultiheadSelfAttentionBlock
from .mlp import MultiLayerPerceptron

class TransformerEncoder(nn.Module):
  def __init__(self,
               embedding_dimension:int=768,
               num_heads:int=12, # Table 1,
               mlp_size:int=3072, # Table 1,
               mlp_dropout:float=0.1,
               att_dropout:int=0):
    super().__init__()
    self.MSA = MultiheadSelfAttentionBlock(
                                          embed_dim=embedding_dimension,
                                          num_heads=num_heads)
    self.MLP = MultiLayerPerceptron(                                    embedding_dimension=embedding_dimension,
    dropout=mlp_dropout,
    mlp_size=mlp_size)

  def forward(self,x):
    x = self.MSA(x) + x # x-> residual connection
    x = self.MLP(x) + x
    return x