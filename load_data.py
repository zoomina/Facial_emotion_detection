from facenet_pytorch import training,fixed_image_standardization
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

class EmoDataset(Dataset):
    def __init__(self, image_path, transform=None):
        super(EmoDataset, self).__init__()
        self.data = datasets.ImageFolder(image_path, transform)
        self.image_path = image_path
        self.transform = transform
        self.data.classes, self.data.class_to_idx = self._find_classes()

    def _find_classes(self):
        im_dir = self.image_path.split("\\")[-2]

        class_to_idx = {
            "angry": 0,
            "disgusted": 1,
            "fearful": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprised": 6

        }

        return im_dir, class_to_idx[im_dir]

    def __getitem__(self, idx):
        img_tensor = self.transform(datasets[idx])
        return (img_tensor)

def data_loader(data_dir, workers, batch_size):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), fixed_image_standardization])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(len(dataset)*0.8)
    dev_size = len(dataset) - train_size
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, dev_size])

    train_loader = DataLoader(
        dataset_train,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    test_loader = DataLoader(
        dataset_train,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    return train_loader, test_loader

def test_loader(data_dir, workers, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), fixed_image_standardization])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil,
        shuffle=True
    )

    return loader