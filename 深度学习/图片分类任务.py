import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

CN, C2I = ['ds', 'hh', 'jy', 'gq', 'bh'], {n: i for i, n in enumerate(['ds', 'hh', 'jy', 'gq', 'bh'])}

def p_img(img):
    img = img.resize((128, 128))
    img = np.array(img).astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.transpose((2, 0, 1))
    return torch.tensor(img)

class TD(Dataset):
    def __init__(self, r):
        self.s = []
        for ln in os.listdir(r):
            lp = os.path.join(r, ln)
            if not os.path.isdir(lp): continue
            for fn in os.listdir(lp):
                if fn.endswith(".jpg"):
                    self.s.append((os.path.join(lp, fn), C2I[ln]))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        p, l = self.s[idx]
        img = Image.open(p).convert("RGB")
        img = p_img(img)
        return img, l

class TsD(Dataset):
    def __init__(self, r):
        self.s = []
        for fn in os.listdir(r):
            if not fn.endswith(".jpg"): continue
            for cn in CN:
                if cn in fn.lower():
                    self.s.append((os.path.join(r, fn), C2I[cn]))
                    break

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        p, l = self.s[idx]
        img = Image.open(p).convert("RGB")
        img = p_img(img)
        return img, l

class CM(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.c = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(CN))
        )

    def forward(self, x):
        x = self.f(x)
        return self.c(x)

td, tsd = TD("cnn/train"), TsD("cnn/test")
tl, tsl = DataLoader(td, batch_size=32, shuffle=True), DataLoader(tsd, batch_size=32, shuffle=False)

mdl, cri, opt = CM(), nn.CrossEntropyLoss(), torch.optim.Adam(mdl.parameters(), lr=1e-4, weight_decay=1e-5)

max_e, es_p, lt, bl, pc = 50, 5, 0.01, float('inf'), 0

ta_l, tl_l = [], []

for e in range(max_e):
    mdl.train()
    tot_l, cor, tot = 0.0, 0, 0

    for imgs, lbs in tl:
        outs = mdl(imgs)
        l = cri(outs, lbs)

        opt.zero_grad()
        l.backward()
        opt.step()

        tot_l += l.item()
        _, pred = torch.max(outs, 1)
        cor += (pred == lbs).sum().item()
        tot += lbs.size(0)

    avg_l = tot_l / len(tl)
    acc = cor / tot
    tl_l.append(avg_l)
    ta_l.append(acc)

    print(f"Epoch {e+1}: Loss={avg_l:.4f}, Train Acc={acc:.4f}")

    if abs(bl - avg_l) < lt:
        pc += 1
        if pc >= es_p:
            print(f"Early stopping triggered at epoch {e+1}.")
            break
    else:
        bl = avg_l
        pc = 0

mdl.eval()
cor, tot = 0, 0
with torch.no_grad():
    for imgs, lbs in tsl:
        outs = mdl(imgs)
        _, pred = torch.max(outs, 1)
        cor += (pred == lbs).sum().item()
        tot += lbs.size(0)

print(f"Test Accuracy: {cor / tot:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(tl_l, label="Train Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ta_l, label="Train Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("training_curves.png")
plt.show()