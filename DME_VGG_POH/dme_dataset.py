from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
from read_json import read_dme_json
import cv2

class Dataset_DME_VGG_PoH(Dataset):
    def __init__(self, json_path, test_mode=False):
        data = read_dme_json(json_path)
        if test_mode:
            data = data[:50]
        
        self.img_path = []
        self.img = []
        self.ground_truth = []

        def load_img_for_vgg(image_path):
            image = Image.open(image_path)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
            image_tensor = preprocess(image)
            return image_tensor

        a = 0
        for dat in data:
            img_path = dat.img_path
            path=  os.path.join('../', img_path)
            # if a == 0:
            #     print(path)
            #     a = 1
            img = load_img_for_vgg(path)
            gt = dat.gt_int
            
            self.img_path.append(img_path)
            self.img.append(img)
            self.ground_truth.append(gt)
        print('loaded data =',len(self.img))
        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        ans = {
            'img_path': self.img_path[idx],
            'img': self.img[idx],
            'ground_truth': self.ground_truth[idx],
        }
        return ans

def main():
    for i in range(0, 12):
        print("'{}': {},".format(str(i), i))


if __name__ =='__main__':
    main()
   
