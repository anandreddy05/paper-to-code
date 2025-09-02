# Vision Transformer (ViT) Implementation

A PyTorch implementation of the Vision Transformer from the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

## 📋 Overview

This project is part of my **papers-to-code** repository series, where I replicate influential research papers in PyTorch. The Vision Transformer revolutionized computer vision by applying the transformer architecture directly to image patches, achieving state-of-the-art results on image classification tasks.

## 🏗️ Architecture

The implementation follows the original ViT architecture:

1. **Patch Embedding**: Images are split into fixed-size patches and linearly embedded
2. **Positional Encoding**: Learnable position embeddings are added to patch embeddings
3. **Transformer Encoder**: Standard transformer encoder layers with multi-head self-attention
4. **Classification Head**: Final linear layer for classification

```
Input Image (224x224x3) 
    ↓
Patch Embedding (16x16 patches) 
    ↓
Position Embedding + [CLS] Token
    ↓
Transformer Encoder (×12 layers)
    ↓
Classification Head
    ↓
Output Classes
```

## 📁 Project Structure

```
Vision_Transformer/
├── train.py                    # Main training script
├── VIT_Paper_Replicating.ipynb # Research notebook with analysis
├── utils/
│   ├── __init__.py
│   ├── dataset.py              # CIFAR-10 data loading and preprocessing
│   ├── plot_curves.py          # Training curve visualization
│   └── train_test_fn.py        # Training and testing functions
└── vit_model/
    ├── __init__.py
    ├── vit.py                  # Main ViT model
    ├── embedding.py            # Patch embedding layer
    ├── encoder.py              # Transformer encoder block
    ├── mlp.py                  # Multi-layer perceptron
    └── msa.py                  # Multi-head self-attention
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib tqdm
```

### Training

```bash
python train.py
```

The script trains on a subset of CIFAR-10 (2000 training samples, 300 test samples) for demonstration purposes.

## 🔧 Model Configuration

Current configuration (ViT-Base/16):

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Patch Size | 16×16 |
| Embedding Dimension | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| MLP Size | 3072 |
| Dropout | 0.1 |

## 📊 Training Details

- **Dataset**: CIFAR-10 (subset for quick training)
- **Optimizer**: Adam (lr=1e-3, weight_decay=0.1)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 10
- **Batch Size**: 32

## 🔍 Key Components

### Patch Embedding (`embedding.py`)
- Converts 2D images into 1D sequences of patch embeddings
- Adds learnable [CLS] token and positional embeddings
- Includes dropout for regularization

### Multi-Head Self-Attention (`msa.py`)
- Implements the attention mechanism from the original transformer
- Uses PyTorch's native MultiheadAttention module
- Includes layer normalization

### MLP Block (`mlp.py`)
- Feed-forward network with GELU activation
- Dropout for regularization
- Layer normalization as per the original paper

### Transformer Encoder (`encoder.py`)
- Combines MSA and MLP blocks
- Implements residual connections
- Stacks multiple layers for deep processing

## 📈 Results

The model trains on CIFAR-10 and tracks:
- Training/Test Loss
- Training/Test Accuracy
- Training time

Results are visualized using the plotting utilities in `utils/plot_curves.py`.

## 🔬 Research Notes

Detailed analysis and experiments can be found in `VIT_Paper_Replicating.ipynb`, including:
- Paper walkthrough and key insights
- Architecture analysis
- Experimental results and comparisons
- Implementation details and design decisions

## 🎯 Future Improvements

- [ ] Add support for different ViT variants (ViT-Large, ViT-Huge)
- [ ] Implement pre-training on larger datasets
- [ ] Add data augmentation strategies
- [ ] Support for different image sizes
- [ ] Model checkpointing and resuming
- [ ] Evaluation on ImageNet

## 📚 References

```bibtex
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## 🤝 Contributing

This is part of my research paper replication series. Feel free to:
- Open issues for bugs or improvements
- Submit pull requests for enhancements
- Suggest other papers to implement

## 📄 License

MIT License - see LICENSE file for details.

---

**Part of the papers-to-code series** 📝 | Replicating influential AI research papers in PyTorch
