import os
import matplotlib.pyplot as plt
import torch

def save_sample(inputs, preds, targets, epoch, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(inputs[0].permute(1,2,0).detach().cpu())
    axs[0].set_title("Input")
    axs[1].imshow(preds[0].permute(1,2,0).detach().cpu())
    axs[1].set_title("Prediction")
    axs[2].imshow(targets[0].permute(1,2,0).detach().cpu())
    axs[2].set_title("Target")
    for ax in axs: ax.axis("off")
    plt.savefig(f"{out_dir}/epoch_{epoch}.png")
    plt.close()
