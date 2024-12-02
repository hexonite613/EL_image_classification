from torchvision import datasets, transforms
import torch
from sklearn.model_selection import KFold,train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.utils.data import ConcatDataset

# to split test_data always same
torch.manual_seed(1004)

import torch.nn as nn
from torch.utils.data import DataLoader

# you can change input size(don't forget to change linear layer!)
custom_transform = transforms.Compose([
    transforms.Resize((100, 200)),
    transforms.ToTensor()
])

train_transforms = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
transforms.RandomRotation(degrees=15),   # ±15도 회전
transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 80~100% 크기에서 랜덤 크롭
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변환
transforms.Resize((100, 200)),
transforms.ToTensor(),  # 이미지를 Tensor로 변환
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

def make_data_loader(args):

    dataset = datasets.ImageFolder(args.data, transform=custom_transform)
    train_data = datasets.ImageFolder(args.data, transform=train_transforms)

    # 데이터셋 병합
    combined_dataset = ConcatDataset([dataset, train_data])

    # 병합된 데이터셋의 타겟 레이블
    combined_labels = dataset.targets + train_data.targets

    # train/val/test split
    train_val_indices, test_indices = train_test_split(
        range(len(combined_dataset)),  # 병합된 데이터셋의 인덱스
        test_size=0.2,
        random_state=42,
        stratify=combined_labels
    )

    # Subset으로 train/val/test 생성
    train_val_dataset = Subset(combined_dataset, train_val_indices)
    test_dataset = Subset(combined_dataset, test_indices)

    # 2. Dataloader for test set
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Divide train and validation set for K-Fold
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    kfold_loaders = []

    fold_idx = 0

    for train_idx, val_idx in kfold.split(train_val_dataset.indices):
        print(f"Training fold {fold_idx + 1}/{kfold.get_n_splits()}")

        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)

        # 4. Create dataloader for train and validation set
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        kfold_loaders.append((train_loader, val_loader))

        fold_idx += 1

    # 5. Return k-fold loaders(test, validation) and test loader
    return kfold_loaders, test_loader
