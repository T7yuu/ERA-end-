import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ThreeWayFeatureAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.projector(x), p=2, dim=-1)


class ThreeWayDecisionLoss(nn.Module):
    def __init__(self, init_alpha=0.3, init_beta=0.8, margin=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))
        self.margin = margin

    def forward(self, anchor, positive, negative, boundary):
        curr_alpha = torch.clamp(self.alpha, min=0.05, max=self.beta.item() - 0.1)
        curr_beta = torch.clamp(self.beta, min=curr_alpha.item() + 0.1, max=2.0)

        dist_pos = torch.norm(anchor - positive, p=2, dim=1)
        dist_neg = torch.norm(anchor - negative, p=2, dim=1)
        dist_bnd = torch.norm(anchor - boundary, p=2, dim=1)

        loss_pos = F.relu(dist_pos - curr_alpha + self.margin)
        loss_neg = F.relu(curr_beta - dist_neg + self.margin)
        loss_bnd = F.relu(curr_alpha - dist_bnd + self.margin) + F.relu(dist_bnd - curr_beta + self.margin)

        return loss_pos.mean() + loss_neg.mean() + loss_bnd.mean()


class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.num_samples = len(features)
        similarity_matrix = torch.matmul(self.features, self.features.T)

        self.pos_idx, self.neg_idx, self.bnd_idx = [], [], []
        for i in tqdm(range(self.num_samples), desc="构建三支元组"):
            sims = similarity_matrix[i]
            sorted_idx = torch.argsort(sims, descending=True)
            self.pos_idx.append(sorted_idx[1:2].item())
            self.neg_idx.append(sorted_idx[-1].item())
            self.bnd_idx.append(sorted_idx[self.num_samples // 2].item())

    def __len__(self): return self.num_samples

    def __getitem__(self, idx): return (self.features[idx], self.features[self.pos_idx[idx]],
                                        self.features[self.neg_idx[idx]], self.features[self.bnd_idx[idx]])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NPZ_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/outputs_medical/medical_features.npz"
    SAVE_PATH = r"E:/PythonD/ERA3/scripts/V9/scripts/3wd_adapter.pth"

    data = np.load(NPZ_PATH)
    dataset = FeatureDataset(data['features'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    adapter = ThreeWayFeatureAdapter().to(device)
    criterion = ThreeWayDecisionLoss().to(device)

    optimizer = torch.optim.AdamW(list(adapter.parameters()) + list(criterion.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    adapter.train()
    criterion.train()
    print(f"\n🚀 开始训练 (包含 Learnable Margins)...")
    for epoch in range(50):
        total_loss = 0
        for anc, pos, neg, bnd in dataloader:
            anc, pos, neg, bnd = anc.to(device), pos.to(device), neg.to(device), bnd.to(device)
            loss = criterion(adapter(anc), adapter(pos), adapter(neg), adapter(bnd))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/50], Loss: {total_loss / len(dataloader):.4f} | Learned α: {criterion.alpha.item():.3f}, β: {criterion.beta.item():.3f}")

    torch.save(adapter.state_dict(), SAVE_PATH)
    print(f"✅ 适配器已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    main()