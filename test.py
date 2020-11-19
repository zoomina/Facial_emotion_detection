import torch
import tqdm

from load_data import *
from model.emo_vgg import *

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    test_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return test_acc

def test(test_dataloader, model, device):
    model.eval()
    answer=[]
    test_acc = 0.0
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            max_vals, max_indices = torch.max(out, 1)
            answer.append(max_indices.cpu().clone().numpy())
            test_acc += calc_accuracy(out, label)
    print(test_acc / (batch_id+1))

if __name__ == "__main__":
    model = VGG().build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_workers = 2
    data_dir = "data/test"

    test_dataloader = test_loader(data_dir, num_workers, batch_size)
    test(test_dataloader, model, device)