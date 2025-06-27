import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os  # Thêm để tạo thư mục
from crnn import CRNN
from dataset import OCRDataset, characters

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hàm collate tùy chỉnh
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, dim=0).to(device)
    labels = [torch.tensor(l, dtype=torch.int64).detach().clone() for l in labels]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.int64)
    labels_1d = torch.cat([l[:length] for l, length in zip(labels, label_lengths)])
    return images, labels_1d, label_lengths

# DataLoader
dataset = OCRDataset(image_dir='data/images', label_file='data/labels.txt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Khởi tạo mô hình
model = CRNN(num_chars=len(characters)).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Tạo thư mục models nếu chưa tồn tại
os.makedirs('models', exist_ok=True)

# Huấn luyện
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels, label_lengths in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        output_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)
        
        loss = criterion(outputs.log_softmax(2), labels, output_lengths, label_lengths)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss is NaN or Inf, skipping this batch.")
            continue
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Lưu mô hình
    torch.save(model.state_dict(), 'models/checkpoint.pt')