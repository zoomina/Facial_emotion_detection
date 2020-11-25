import torch
import matplotlib.pyplot as plt

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def result_graph(train_summary, epoch, batch_size):
    plt.subplot(1, 3, 1)
    plt.plot(range(len(train_summary["loss"])), train_summary["loss"], label="train_loss")
    plt.subplot(1, 3, 2)
    plt.plot(range(len(train_summary["acc"])), train_summary["acc"], label="train_acc")
    plt.subplot(1, 3, 3)
    plt.plot(range(len(train_summary["eval"])), train_summary["eval"], label="eval_acc")
    plt.savefig("result/train_summary_epoch{}_batch{}.jpg".format(epoch, batch_size))