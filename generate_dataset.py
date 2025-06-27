import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Cấu hình
output_dir = 'data/images'
labels_file = 'data/labels.txt'
os.makedirs(output_dir, exist_ok=True)

# Tập câu tiếng Việt mẫu
texts = [
    'xin chào', 'việt nam', 'hà nội', 'sài gòn', 'chúc mừng',
    'cảm ơn', 'tạm biệt', 'học tập', 'công việc', 'gia đình',
    'thành phố', 'quốc gia', 'đất nước', 'tự hào', 'lịch sử'
]

# Load đúng file font cùng cấp với script
font_path = "Kujang Ciung Basyri.ttf"

# Hàm thêm nhiễu
def add_noise(img, amount=10):
    np_img = np.array(img)
    noise = np.random.randint(0, amount, np_img.shape, dtype='uint8')
    np_img = np.clip(np_img + noise, 0, 255)
    return Image.fromarray(np_img)

# Sinh dữ liệu
try:
    font = ImageFont.truetype(font_path, 24)
except Exception as e:
    print(f"Lỗi tải font {font_path}: {e}")
    font = ImageFont.load_default()

with open(labels_file, 'w', encoding='utf-8') as f:
    idx = 0
    for text in texts:
        for _ in range(50):  
            img = Image.new('L', (128, 32), color='white')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(font_path, random.randint(18, 28))
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pos = ((128 - w) // 2, (32 - h) // 2)
            draw.text(pos, text, fill='black', font=font)

            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            if random.random() < 0.3:
                img = add_noise(img, amount=20)

            img_name = f'img_{idx}.png'
            img.save(os.path.join(output_dir, img_name))
            f.write(f"{img_name}\t{text}\n")
            idx += 1

print(f"✅ Đã sinh {idx} ảnh vào '{output_dir}' và file nhãn '{labels_file}'")
