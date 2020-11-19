from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os

from model.emo_vgg import *
from load_data import *

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def train(train_dataloader, dev_dataloader, model, device, batch_size, optimizer, criterion, epoch):
    max_grad_norm = 1
    log_interval = 200
    train_summary = {"loss": [], "acc": [], "eval":[]}
    for e in range(epoch):
        train_acc = 0.0
        test_acc = 0.0
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        step_summary = {"loss" : 0, "acc" : 0, "eval" : 0}
        model.train()
        for batch_id, (img, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = torch.stack(img).to(device)
            label = torch.tensor(label).long().to(device)
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            step_summary["loss"] += loss.data
            step_summary["acc"] += train_acc / (batch_id + 1)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        train_summary["loss"].append(step_summary["loss"] / batch_size)
        train_summary["acc"].append(step_summary["acc"] / batch_size)

        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
        model.eval()  # 모델 평가 부분
        for batch_id, (img, label) in enumerate(dev_dataloader):
            img = torch.stack(img).to(device)
            label = torch.FloatTensor(label).to(device)
            out = model(img)
            test_acc += calc_accuracy(out, label)
            step_summary["eval"] += test_acc / (batch_id + 1)
        train_summary["eval"].append(step_summary["eval"] / batch_size)
        print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))


    if not os.path.exists("./result"):
        os.mkdir("./result")
    plt.subplot(1, 3, 1)
    plt.plot(range(len(train_summary["loss"])), train_summary["loss"], label="train_loss")
    plt.subplot(1, 3, 2)
    plt.plot(range(len(train_summary["acc"])), train_summary["acc"], label="train_acc")
    plt.subplot(1, 3, 3)
    plt.plot(range(len(train_summary["eval"])), train_summary["eval"], label="eval_acc")
    plt.imsave("result/train_summary_epoch{}_batch{}.jpg".format(epoch, batch_size))
    torch.save(model.state_dict(), 'result/epoch{}_batch{}.pt'.format(epoch, batch_size))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG(num_classes=7).to(device)
    batch_size = 32
    num_workers = 1
    epoch = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    data_dir = "data/test"

    train_dataloader, dev_dataloader = data_loader(data_dir, num_workers, batch_size)

    train(train_dataloader, dev_dataloader, model, device, batch_size, optimizer, criterion, epoch)
