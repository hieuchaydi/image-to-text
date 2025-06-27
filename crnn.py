import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),    # (B, 64, 32, W)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),           # (B, 64, 16, W//2)

            nn.Conv2d(64, 128, 3, 1, 1), # (B, 128, 16, W//2)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),           # (B, 128, 8, W//4)

            nn.Conv2d(128, 256, 3, 1, 1),# (B, 256, 8, W//4)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),           # (B, 256, 4, W//8)

            nn.Conv2d(256, 512, 3, 1, 1),# (B, 512, 4, W//8)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(4, 1),           # (B, 512, 1, W//8)
        )

        self.rnn = nn.LSTM(512, rnn_hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size * 2, num_chars + 1)  # +1 cho CTC blank

    def forward(self, x):
        x = self.cnn(x)                         # (B, 512, 1, W')
        assert x.shape[2] == 1, f"Chiều cao sau CNN phải là 1, nhưng nhận được {x.shape[2]}"
        
        x = x.squeeze(2)                       # (B, 512, W')
        x = x.permute(0, 2, 1)                # (B, W', 512)
        
        x, _ = self.rnn(x)                    # (B, W', H*2)
        x = self.fc(x)                        # (B, W', num_chars + 1)
        
        x = x.permute(1, 0, 2)                # (W', B, num_chars + 1) theo chuẩn CTC
        return x
