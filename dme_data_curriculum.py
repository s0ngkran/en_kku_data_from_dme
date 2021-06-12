from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

class DMEDataset(Dataset):
    def make_grayscale(self, img_path):
        image_cv2 = cv2.imread(img_path)
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        return gray
    def preprocessing_for_vgg(self, image_path):
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = preprocess(image)
        return image_tensor


    def __init__(self, json, test_mode=False):
        if test_mode:
            json = json[:50]
        self.images = []
        self.gts = []
        self.gtl = []
        self.gts_mask = []
        self.gtl_mask = []
        self.covered_point = []
        self.covered_link = []
        self.image_path = []

        for index, dat in enumerate(json):
            # print('loading... ', index, len(json)-1)
            img_path = dat['path']
            gts_path = dat['gts']
            gts_mask_path = dat['gts_mask']
            gtl_path = dat['gtl']
            gtl_mask_path = dat['gtl_mask']
            covered_point = dat['covered_point']
            covered_link = dat['covered_link']

            
            self.image_path.append(dat['path'])
            # self.images.append(self.make_grayscale(img_path))
            self.images.append(self.preprocessing_for_vgg(img_path))
            self.gts.append(torch.load(gts_path).float())
            self.gtl.append(torch.load(gtl_path).float())
            self.gts_mask.append(torch.load(gts_mask_path).float())
            self.gtl_mask.append(torch.load(gtl_mask_path).float())

            # you need to solve this
            covered_point = np.array(covered_point).astype(np.bool)
            ################################################################
            
            self.covered_point.append(torch.tensor(covered_point, dtype=torch.bool))
            self.covered_link.append(torch.tensor(covered_link, dtype=torch.bool))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ans = {
            'image_path': self.image_path[idx],
            'image': self.images[idx],
            'gts': self.gts[idx],
            'gtl': self.gtl[idx],
            'gts_mask': self.gts_mask[idx],
            'gtl_mask': self.gtl_mask[idx],
            'covered_point': self.covered_point[idx],
            'covered_link': self.covered_link[idx],
        }
        return ans
