import os
from PIL import Image, ImageDraw, ImageFont
texts = ['xin chào', 'việt nam', 'hà nội', 'sài gòn', 'chúc mừng', 'cảm ơn', 'tạm biệt', 'học tập', 'công việc', 'gia đình']
os.makedirs('data/images', exist_ok=True)
font = ImageFont.truetype("arialuni.ttf", 24)
with open('data/labels.txt', 'w', encoding='utf-8') as f:
    for i, text in enumerate(texts):
        img = Image.new('RGB', (128, 32), 'white')
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text, font=font)
        draw.text(((128-w)//2, (32-h)//2), text, font=font, fill='black')
        img.save(f"data/images/img{i+1}.png")
        f.write(f"img{i+1}.png\t{text}\n")