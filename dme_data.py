from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
class DMEDataset(Dataset):
    def make_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def __init__(self, json, test_mode=False):
        if test_mode:
            json = json[:50]
        self.images = []
        self.gts = []
        self.gtl = []
        self.covered_point = []
        self.covered_link = []
        self.image_path = []

        for index, dat in enumerate(json):
            # print('loading... ', index, len(json)-1)
            img_path = dat['path']
            gts_path = dat['gts']
            gtl_path = dat['gtl']
            covered_point = dat['covered_point']
            covered_link = dat['covered_link']

            
            img = cv2.imread(img_path)
            self.image_path.append(dat['path'])
            self.images.append(self.make_grayscale(img))
            self.gts.append(torch.load(gts_path).float())
            self.gtl.append(torch.load(gtl_path).float())

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
            'covered_point': self.covered_point[idx],
            'covered_link': self.covered_link[idx],
        }
        return ans
