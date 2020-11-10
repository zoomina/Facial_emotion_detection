import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from model.emo_vgg import *

PATH = "./result"

def train(X, y, model, device, batch_size, optimizer, criterion, epoch):
    X = torch.tensor(X, dtype=torch.float).to(device)
    y = torch.tensor(y, dtype=torch.float).to(device)
    loss = criterion(X, y)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    train_summary = {"loss": []}
    for i in range(epoch):
        start_time = time.time()
        step_size = X.shape[0] // batch_size
        step_summary = {"loss": 0}
        for step in range(step_size):
            batch_mask = range(i * batch_size, (i + 1) * batch_size)
            optimizer.zero_grad()  # zero the gradient buffers
            output = model(X[batch_mask])
            loss = loss(output, y[batch_mask])
            loss.backward()
            optimizer.step()  # Does the update
            step_summary["loss"] += loss.data

        scheduler.step()
        print("epoch [{:3}/{:3}] time [{:6.4f}s] loss [{:.7f}]".format(i + 1, epoch, time.time() - start_time,
                                                                       step_summary["loss"] / step_size))
        train_summary["loss"].append(step_summary["loss"] / step_size)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # ...
    }, PATH)

    plt.plot(range(len(train_summary["loss"])), train_summary["loss"])
    plt.savefig("./result/loss.png")

def main():
    model = VGG.build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()