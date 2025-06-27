import cv2
import torch
from torch.utils.data import Dataset
import os

# Từ điển ký tự tiếng Việt
characters = list(" 0123456789abcdefghijklmnopqrstuvwxyzáàảãạâấầẩẫậăắằẳẵặđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ")
idx_to_char = {i + 1: c for i, c in enumerate(characters)}
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}


class OCRDataset(Dataset):
    def __init__(self, image_dir, label_file):
        self.image_dir = image_dir
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = [line.strip().split('\t') for line in f]
        self.image_paths = [os.path.join(image_dir, img_name) for img_name, _ in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx][1]
        
        # Tiền xử lý ảnh
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        img = cv2.resize(img, (128, 32))  # Chuẩn hóa kích thước
        img = img / 255.0  # Chuẩn hóa pixel
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Thêm channel

        # Mã hóa nhãn
        label_encoded = [char_to_idx[c] for c in label if c in char_to_idx]
        label_tensor = torch.tensor(label_encoded, dtype=torch.int64)

        return img, label_tensor, len(label_encoded)