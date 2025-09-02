from utils.train_test_fn import train
from utils.dataset import train_dataloader,test_dataloader,class_names,device
from utils.plot_curves import plot_predictions

from vit_model.vit import VitTransformer
from torch import optim, nn

if __name__ == "__main__":
    print("Vision Transformer Training...")
    model = VitTransformer(in_channels=3,
                    embedding_dimension=768,
                    num_transformer_layers=12,
                    patch_size=16,
                    image_size=224,
                    num_heads=12,
                    mlp_size=3072,
                    mlp_dropout=0.1,
                    att_dropout=0,
                    embedding_dropout=0.1,
                    num_classes=len(class_names))
    LR = 1e-3
    weight_decay = 0.1
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = optim.Adam(model.parameters(), lr=LR,weight_decay=weight_decay)
    EPOCHS = 10 


    from timeit import default_timer as timer
    start_time = timer()
    results = train(num_epochs=EPOCHS,
                    model=model,
                    train_loader=train_dataloader,
                    test_loader=test_dataloader,
                    loss_fn=LOSS_FN,
                    optimizer=OPTIMIZER,
                    device=device)
    end_time = timer()
    print(f"Results:\n{results}\n")

    print(f"Time taken: {end_time-start_time:.3f} seconds")