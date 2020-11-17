from torch.optim.lr_scheduler import StepLR
import tqdm
import os

from model.emo_vgg import *
from load_data import *

PATH = "./result"

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def train(train_dataloader, dev_dataloader, model, device, batch_size, optimizer, criterion, epoch):
    for e in range(epoch):
        train_acc = 0.0
        test_acc = 0.0
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = criterion(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
        model.eval()  # 모델 평가 부분
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(dev_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))


    if not os.path.exists("./result"):
        os.mkdir("./result")
    torch.save(model.state_dict(), 'result/epoch{}_batch{}.pt'.format(epoch, batch_size))

def main():
    model = VGG().build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train_dataloader = data_loader("data/train", workers=2, batch_size=64)