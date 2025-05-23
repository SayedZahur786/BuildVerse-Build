import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class SkinDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe  
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_name"])
        
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row.iloc[1:].values.astype(float), dtype=torch.float32)
        
        return image, labels

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_dataloaders(label_file, image_dir, batch_size=32):
    transform = get_transforms()

    df = pd.read_excel(label_file)

    if df.iloc[:, 1:].sum().min() < 2:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, 1:])

    train_dataset = SkinDataset(train_df, image_dir, transform=transform)
    val_dataset = SkinDataset(val_df, image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
