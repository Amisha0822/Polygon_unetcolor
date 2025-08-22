import torch
from unet import UNet
from dataset import PolygonDataset
from utils import save_sample

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load("unet.pth", map_location=device))
    model.eval()

    dataset = PolygonDataset(size=5)
    img, mask = dataset[0]

    with torch.no_grad():
        pred = model(torch.tensor(img).unsqueeze(0).to(device))
    
    save_sample(torch.tensor(img).unsqueeze(0), pred, torch.tensor(mask).unsqueeze(0), "inference")

if __name__ == "__main__":
    run_inference()
