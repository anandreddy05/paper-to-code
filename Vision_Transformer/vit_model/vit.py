from torch import nn
from .encoder import TransformerEncoder

from .embedding import ViTEmbedding

class VitTransformer(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 embedding_dimension:int=768,
                 num_transformer_layers:int=12,
                 patch_size:int=16,
                 image_size:int=224,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 att_dropout:int=0,
                 embedding_dropout:float=0.1,
                 num_classes:int=1000):
        super().__init__()

        assert image_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {image_size}, patch size: {patch_size}."

        self.embedding = ViTEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dimension=embedding_dimension,
            image_size=image_size,
            embedding_dropout=embedding_dropout
        )

        self.encoder = nn.Sequential(
            *[TransformerEncoder(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                att_dropout=att_dropout
            ) for _ in range(num_transformer_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dimension),
            nn.Linear(in_features=embedding_dimension, out_features=num_classes)
        )

    def forward(self, x):
      x_embed = self.embedding(x)         # [B, 197, 768]
      x_encoded = self.encoder(x_embed)   # [B, 197, 768]
      cls_token = x_encoded[:, 0]         # [B, 768]
      x_classified = self.classifier(cls_token) # [B, num_classes]
      return x_classified
