import torch
import cv2
from crnn import CRNN
from dataset import characters, idx_to_char
from decode import decode_output

def predict_image(image_path, model_path='models/checkpoint.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(num_chars=len(characters)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.convertScaleAbs(img_gray, alpha=1.5, beta=0)

    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_cropped = img_gray[y:y+h, x:x+w]
    else:
        img_cropped = img_gray

    img_resized = cv2.resize(img_cropped, (128, 32), interpolation=cv2.INTER_AREA)
    _, img_resized = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_resized = img_resized / 255.0

    img = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
    text = decode_output(output)
    return text[0] if text else "Không nhận diện được"