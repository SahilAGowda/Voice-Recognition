import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from model import EmbeddingNet, save_model
from server import SAMPLE_RATE, N_MFCC, extract_features


class TripletAudioDataset(Dataset):
    def __init__(self, root: Path, sr: int = SAMPLE_RATE):
        self.root = Path(root)
        self.sr = sr
        # Expect folder structure: root/person_id/*.wav; root/noise/*.wav optionally
        self.items = []
        for p in sorted(self.root.glob('*')):
            if p.is_dir():
                wavs = list(p.glob('*.wav'))
                if len(wavs) >= 2:
                    self.items.append((p.name, wavs))
        self.labels = [lbl for lbl, _ in self.items]

        # Build index by label
        self.by_label = {lbl: wavs for lbl, wavs in self.items}
        self.labels_set = list(self.by_label.keys())

    def __len__(self):
        return 10000  # virtual length for sampling

    def load_wav(self, path: Path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(y) == 0:
            y = np.zeros(self.sr, dtype=np.float32)
        y = y / (np.max(np.abs(y)) + 1e-9)
        return y.astype(np.float32)

    def __getitem__(self, idx):
        # Sample anchor label
        a_lbl = random.choice(self.labels_set)
        pos_files = self.by_label[a_lbl]
        if len(pos_files) < 2:
            return self.__getitem__(idx + 1)
        a_path, p_path = random.sample(pos_files, 2)
        # Negative from another label
        n_lbl = random.choice([l for l in self.labels_set if l != a_lbl])
        n_path = random.choice(self.by_label[n_lbl])

        def to_feat(path):
            y = self.load_wav(path)
            f = extract_features(y)
            return f

        a = to_feat(a_path)
        p = to_feat(p_path)
        n = to_feat(n_path)
        return (
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(n, dtype=torch.float32),
        )


def train(args):
    ds = TripletAudioDataset(Path(args.data))
    # Infer input dim from one sample
    a, p, n = ds[0]
    in_dim = a.numel()
    model = EmbeddingNet(in_dim=in_dim, emb_dim=args.emb)
    model.train()

    def collate(batch):
        a = torch.stack([b[0] for b in batch])
        p = torch.stack([b[1] for b in batch])
        n = torch.stack([b[2] for b in batch])
        return a, p, n

    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=collate)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    triplet = nn.TripletMarginLoss(margin=args.margin, p=2.0)

    for step, (a, p, n) in enumerate(dl, 1):
        # Normalize inputs (z-score) before MLP to help training
        def norm(x):
            m = x.mean(dim=0, keepdim=True)
            s = x.std(dim=0, keepdim=True) + 1e-6
            return (x - m) / s

        a = norm(a); p = norm(p); n = norm(n)
        za = model(a)
        zp = model(p)
        zn = model(n)
        loss = triplet(za, zp, zn)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")
        if step >= args.steps:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to dataset root (folders per speaker)')
    ap.add_argument('--out', default='backend/data/embedding_net.pt')
    ap.add_argument('--emb', type=int, default=128)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--steps', type=int, default=1000)
    args = ap.parse_args()
    train(args)
