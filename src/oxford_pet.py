import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class OxfordPetDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'annotations', 'trimaps')
        self.transform = transform
        self.mode = mode

        if mode == 'train':
            split_file = './dataset/train.txt'
        elif mode == 'val':
            split_file = './dataset/val.txt'
        else:
            raise ValueError("mode must be 'train' or 'val'")

        with open(split_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        img_path = os.path.join(self.image_dir, file_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, file_name + '.png')

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize((256, 256))
        mask = mask.resize((256, 256), Image.NEAREST)

        if self.mode == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.randint(-15, 15)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image)

        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32))

        binary_mask = torch.zeros_like(mask_tensor)
        binary_mask[mask_tensor == 1.0] = 1.0
        binary_mask[mask_tensor == 2.0] = 0.0
        binary_mask[mask_tensor == 3.0] = 0.0

        binary_mask = binary_mask.unsqueeze(0)

        return image_tensor, binary_mask


def get_dataloaders(data_dir, batch_size=16):
    train_dataset = OxfordPetDataset(data_dir, mode='train')
    val_dataset = OxfordPetDataset(data_dir, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader