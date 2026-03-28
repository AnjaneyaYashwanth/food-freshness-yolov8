from dataset_loader import DualStreamDataset
import torch

dataset = DualStreamDataset("data/processed", split="train")

print("Total samples:", len(dataset))

rgb, edge, lbp, label = dataset[0]

print("RGB shape:", rgb.shape)
print("Edge shape:", edge.shape)
print("LBP shape:", lbp.shape)
print("Label:", label)

x = torch.cat([rgb, edge, lbp], dim=0)
print("Stacked shape:", x.shape)
