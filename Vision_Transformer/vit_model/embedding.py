import torch
from torch import nn


class ViTEmbedding(nn.Module):
  """ Turns a 2D image into 1D sequence of learnable embedding vector

  Args:
   in_channels (int): Number of colour channels for the input image. patch_size (int): Size of patches to convert input image into. out_channels (int): Size of embedding to turn each patch images into embeddings.

   """
  # Initialize hyperparameters
  def __init__(self,
               in_channels:int=3,
               patch_size:int=16,
               embedding_dimension:int=768,
               image_size:int=224,
               embedding_dropout:float=0.1):
    super().__init__()
    assert image_size % patch_size == 0 ,f"Image size must be divisible by patch size, image size: {image_size}, patch size: {patch_size}."

    # Converting Image into embedding patches
    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dimension,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)

    num_patches = (image_size * image_size)//patch_size**2

    # Flattenning the embeddings
    self.flatten = nn.Flatten(start_dim=2,
                              end_dim=3)


    # Learnable [CLS] token
    self.class_token = nn.Parameter(torch.randn(1,1,embedding_dimension))

    # Learnable positional embeddings
    self.positional_embeddings = nn.Parameter(torch.randn(1,num_patches+1,embedding_dimension))

    # Embedding and Patch Dropout
    self.embedding_dropout = nn.Dropout(p=embedding_dropout)

  def forward(self,x):
    B = x.size(0) # batch_size
    # [B,C,H,W] -> [B,D,H//ps,W//ps]
    x_pathced = self.patcher(x)
    # [B,D,H//ps,W//ps] -> [B,D,N]
    x_flattened = self.flatten(x_pathced)
    # [B,D,N] -> [B,N,D]
    x = x_flattened.permute(0,2,1)
    # prepend class token => [B,1,D]
    cls_token = self.class_token.expand(B,-1,-1)
    x_with_cls = torch.cat((cls_token,x),dim=1)
    # add positional embeding => [B,N+1,D]
    x_out = x_with_cls + self.positional_embeddings
    x_out = self.embedding_dropout(x_out)
    return x_out