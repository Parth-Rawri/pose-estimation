# datasete.py
import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class RGBDataset(Dataset):
    def __init__(self, root_dir, ground_truth=False, transform=True):
        self.root_dir = root_dir
        self.ground_truth = ground_truth
        self.transform = transform
        filenames = os.listdir(os.path.join(self.root_dir, 'rgb'))
        self.dataset_len = len(filenames)
    
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        rgb_img = cv2.imread(os.path.join(self.root_dir, 'rgb', f"{idx}_rgb.png"))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        if self.ground_truth:
            label = cv2.imread(os.path.join(self.root_dir, 'gt', f"{idx}_gt.png"), -1)
            sample = {'input':rgb_img, 'target':label}
        else:
            sample = {'input':rgb_img}

        if self.transform:
            mean_rgb = [0.722, 0.751, 0.807]
            std_rgb = [0.171, 0.179, 0.197]
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean_rgb, std=std_rgb),])
            sample['input'] = transform(sample['input'])
        return sample


