import argparse

import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel
import torch.nn.init as init
import torch
import torch.nn as nn # edited
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau

def acc(pred, label):
    pred = (pred >= 0.5).float()
    return torch.sum(pred == label).item()


def init_weights(model):
    if isinstance(model, nn.Linear):
        init.xavier_uniform_(model.weight)
        if model.bias is not None:
            init.zeros_(model.bias)

def train(args, k_fold_loader, model):
    criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)

    best_val_acc = 0

    try:
        checkpoint = torch.load(f'{args.save_path}/model(learn).pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Previously saved model loaded successfully.")
    except FileNotFoundError:
        init_weights(model)
        print("No previously saved model found, training from scratch.")

    for fold, (train_loader, val_loader) in enumerate(k_fold_loader):
        print(f"[Fold {fold + 1}] Training the model...")
    
        for epoch in range(args.epochs):
            train_losses = [] 
            train_acc = 0.0
            total=0
            print(f"[Epoch {epoch+1} / {args.epochs}]")
            
            model.train()
            pbar = tqdm(train_loader)
            for i, (x, y) in enumerate(pbar):
                image = x.to(args.device)
                label = y.to(args.device).float().squeeze()         
                optimizer.zero_grad()

                
                output = model(image).squeeze()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                total += label.size(0)

                train_acc += acc(output, label)

            print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")

            epoch_train_loss = np.mean(train_losses)
            epoch_train_acc = train_acc/total
            
            print(f'Epoch {epoch+1}') 
            print(f'train_loss : {epoch_train_loss}')
            print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))

            model.eval()
            val_losses = []
            val_acc = 0.0
            val_total = 0
            
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc='Validation'):
                    image = x.to(args.device)
                    label = y.to(args.device).float().squeeze()
                    label = label.squeeze()
                    
                    output = model(image).squeeze()
                    loss = criterion(output, label)
                    
                    val_losses.append(loss.item())
                    val_total += label.size(0)

                    preds = (output >= 0.5).float()
                    val_acc += (preds == label).sum().item()
            
            # Print epoch results
            epoch_train_loss = np.mean(train_losses)
            epoch_train_acc = 100 * train_acc / total
            epoch_val_loss = np.mean(val_losses)
            epoch_val_acc = 100 * val_acc / val_total
            print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

            scheduler.step(epoch_val_loss)
            
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), f'{args.save_path}/model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, f'{args.save_path}/model(learn).pth')
                print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
    print(f"Last best model with validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2024 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='test_image/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 1
    
    # hyperparameters
    args.epochs = 30
    args.learning_rate = 0.1
    args.batch_size = 256
    args.k_folds = 5

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")

    # Make Data loader and Model
    k_fold_loader,_ = make_data_loader(args)
    
    # torchvision model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1)
    )
    model.to(device)
    print(model)


    # Training The Model
    train(args, k_fold_loader, model)