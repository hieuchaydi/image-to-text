import torch
from dataset import idx_to_char  # Import từ thư mục gốc

def decode_output(output):
    output = output.permute(1, 0, 2)
    output = output.softmax(2)
    output = output.argmax(2)
    decoded = []
    for seq in output:
        seq = seq.cpu().numpy()
        text = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != 0:
                text.append(idx_to_char.get(idx, ''))
            prev = idx
        decoded.append(''.join(text))
    return decoded