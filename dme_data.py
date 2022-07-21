import json
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch


class DMEDataset(Dataset):
    def make_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    def read_keypoint_from_json(json):
        return

    def __init__(self, json=None, json_path=None, test_mode=False):
        false = False
        true = True
        a  =     {"path": "testing_set/0IMG_1789.jpg", "keypoint": [[271.8, 216.8, "0", "0"], [234.8, 185.30000000000004, "0", "0"], [222.3, 173.8, "0", "0"], [210.3, 161.8, "1", "1"], [270.8, 183.29999999999998, "0", "0"], [256.3, 150.3, "0", "0"], [252.3, 135.8, "0", "0"], [250.3, 120.8, "0", "0"], [246.3, 109.3, "0", "0"], [269.3, 148.3, "0", "0"], [271.3, 131.3, "0", "0"], [272.3, 114.8, "0", "0"], [272.8, 97.3, "0", "0"], [282.3, 152.3, "0", "0"], [284.8, 134.3, "0", "0"], [284.8, 116.3, "0", "0"], [287.3, 101.80000000000001, "0", "0"], [294.3, 161.3, "0", "0"], [301.3, 146.8, "0", "0"], [305.3, 136.8, "0", "0"], [313.3, 124.30000000000001, "0", "0"], [161.29999999999998, 160.8, "0", "0"], [180.8, 160.8, "0", "0"], [193.8, 160.8, "0", "0"], [206.3, 158.8, "0", "0"]], "hand_side": "L", "gt": "2\n", "user": "DMW", "status": "testing", "gts": "testing_set/0IMG_1789.jpg.gts", "gtl": "testing_set/0IMG_1789.jpg.gtl", "covered_point": ["0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"], "covered_link": [false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]}
        for k,v in a.items():
            print(k, v)
        return
        if json_path is not None:
            with open(json_path, 'r') as f:
                json = json.load(f)
            json = json
            ### return new json
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

def main():
    json = 'validation_json_with_covered'
    with open(json, 'r') as f:
        json_obj = json.load(f)
    data = DMEDataset(json_obj)
    print('n_data=', len(data))

def dummy_init():
    data = DMEDataset()

if __name__ == '__main__':
    dummy_init()