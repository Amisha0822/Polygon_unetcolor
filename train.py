import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from dataset import PolygonDataset
from utils import save_sample

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=3).to(device)
    dataset = PolygonDataset(size=500)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # save sample every epoch
        save_sample(imgs, preds, masks, epoch)

    torch.save(model.state_dict(), "unet.pth")
    print("Training finished, model saved.")

if __name__ == "__main__":
    train()
