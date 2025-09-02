from torch import nn
import torch

class MultiLayerPerceptron(nn.Module):
  """
  Utilizes torch.nn.Sequential Layer to build MLP and torch.LayerNorm to replicate equation 3 of ViT paper
  """
  def __init__(self,
               embedding_dimension:int=768,
               mlp_size:int=3072,
               dropout:float=0.1,
               ):
    super().__init__()

    # Layer Norm
    self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dimension)

    # Create Multi Layer Perceptron
    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dimension,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embedding_dimension),
        nn.Dropout(p=dropout)
    )

  def forward(self,x):
    x_normalized = self.layer_norm(x)
    return self.mlp(x_normalized)