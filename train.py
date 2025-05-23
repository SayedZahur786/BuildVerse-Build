import torch
import torch.optim as optim
import torch.nn as nn
from model import SkinCNN
from dataset import get_dataloaders

label_file = "label.xlsx"
image_dir = "images"

train_loader, val_loader = get_dataloaders(label_file, image_dir, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 30 
model = SkinCNN(num_classes).to(device)

criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "skin_model.pth")
print("Model saved successfully.")
