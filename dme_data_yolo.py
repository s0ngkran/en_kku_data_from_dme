from torch.utils.data import Dataset
import cv2
import torch

class DMEDataset(Dataset):
    def make_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def __init__(self, json, test_mode=False):
        if test_mode:
            json = json[:40]
        self.image = []
        self.gt = []
        for index, dat in enumerate(json):
            # print('loading... ', index, len(json)-1)
            img_path = dat['path']
            gt = dat['gt']

            img = cv2.imread(img_path)
            self.image.append(self.make_grayscale(img))
            self.gt.append(gt)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        ans = {
            'image': self.image[idx],
            'gt':self.gt[idx],
        }
        return ans
