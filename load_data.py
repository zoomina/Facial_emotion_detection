from facenet_pytorch import training
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class EmoDataset(Dataset):
    def __init__(self, image_path, transform=None):
        super(EmoDataset, self).__init__()
        self.data = datasets.ImageFolder(image_path, transform)
        self.data.classes, self.data.class_to_idx = self._find_classes(image_path)
        self.image_path = image_path

    def _find_classes(self, dir):
        im_dir = self.image_path.split("\\")[-2]
        class_to_idx = {
            "angry": 0,
            "disgusted": 1,
            "fearful": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprised" : 6
        }
        return im_dir, class_to_idx[im_dir]

def data_loader(data_dir, workers, batch_size):
    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
    ]

    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    return loader
