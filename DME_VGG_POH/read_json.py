import json
import os
import cv2

from iou_cal import BBoxIOU
try: 
    import matplotlib.pyplot as plt
except:
    print('no matplot')
    pass
import numpy as np

class Point:
    def __init__(self, data=None, x=None, y=None):
        if data is not None and len(data) == 4:
            self.x, self.y, _, __ = data
        else:
            self.x = x
            self.y = y
        self.dist = None
        self.tag = None
        self.index = None
    
    def plot(self, color='or'):
        plt.plot(self.x, self.y, color)
    
    def vector(self, p):
        x = p.x - self.x
        y = p.y - self.y
        return x, y

    def distance(self, p, w=1, h=1):
        dist = ((p.x  - self.x)*w)**2 + ((p.y - self.y)*h)**2
        dist = dist**0.5
        return dist

class DMEDataFromJson:
    def __init__(self, data):
        self.dataset_name = 'DME data'
        self.img_path = data['path']
        self.img_name = self.img_path.split('/')[-1]
        self.keypoint = data['keypoint']
        self.hand_side = data['hand_side']
        self.gt = data['gt'].strip()
        self.gt_int = int(self.gt)
        self.user = data['user']
        self.status = data['status'] # training, ...

        # unused for vgg
        self.gts = data['gts']
        self.gtl = data['gtl']
        self.covered_point = data['covered_point']
        self.covered_link = data['covered_link']

        def read_hand(data):
            hand = []
            for point_data in data:
                hand.append(Point(point_data))
            if hand == []:
                print('empty hand')
            return hand
        self.hand = read_hand(self.keypoint)
        self.n_keypoint = len(self.hand)
    
        # extract keypoint
        self.palm_hand = self.hand[:21]
        self.pointing_hand = self.hand[21:]
        self.hands = [self.palm_hand, self.pointing_hand]
        self.pointing_tip = self.hand[-1]
        assert len(self.pointing_hand) == 4
        assert len(self.palm_hand) == 21

        # find bbox of pointing_hand and palm_hand
        min_x, max_x, min_y, max_y = self.find_bbox(self.pointing_hand)
        xc = (min_x + max_x)*0.5
        yc = (min_y + max_y)*0.5
        self.center_of_pointing_hand = Point(x=xc, y=yc)
        min_x, max_x, min_y, max_y = self.find_bbox(self.palm_hand)
        xc = (min_x + max_x)*0.5
        yc = (min_y + max_y)*0.5
        self.center_of_palm_hand = Point(x=xc, y=yc)
        self.bbox_pointing = BBoxIOU(self.find_bbox(self.pointing_hand))
        self.bbox_palm = BBoxIOU(self.find_bbox(self.palm_hand))
        

    def find_bbox(self, points):
        points.sort(key=lambda a: a.x)
        min_x = points[0].x
        max_x = points[-1].x
        points.sort(key=lambda a: a.y)
        min_y = points[0].y
        max_y = points[-1].y
        return min_x, max_x, min_y, max_y

    def plot(self, root_dir=None):
        if root_dir is not None:
            self.img_path = os.path.join(root_dir, self.img_path)
        img = cv2.imread(self.img_path)
        w, h, channel = img.shape


        def plot_line(p1, p2, w=w, h=h, color='-y', is_scaled=False):
            assert type(p1) == Point
            if is_scaled:
                w = 1
                h = 1
            x1, y1 = p1.x * w, p1.y*h
            x2, y2 = p2.x * w, p2.y*h
            plt.plot((x1, x2), (y1, y2), color)
        
        def plot_bbox(bbox):
            min_x, max_x, min_y, max_y = bbox
            plot_line(Point(x=min_x, y=min_y), Point(x=max_x, y=min_y), is_scaled=True)
            plot_line(Point(x=min_x, y=max_y), Point(x=max_x, y=max_y), is_scaled=True)
            plot_line(Point(x=min_x, y=min_y), Point(x=min_x, y=max_y), is_scaled=True)
            plot_line(Point(x=max_x, y=min_y), Point(x=max_x, y=max_y), is_scaled=True)

        # img = cv2.flip(img, 1)
        plt.imshow(img[:, :, (2, 1, 0)])
        plt.title('gt='+str(self.gt))
        for i, p in enumerate(self.hand):
            # i <= 20 is palm pose
            # i == 21, 22, 23, 24 is pointing pose
            if i <= 20:
                color = 'or'
            else:
                color = 'ob'
            
            if i == 24:
                color = 'og'
            print(p.x)
            plt.plot(p.x, p.y, color)
            # plot_line(hand[0], hand[2])
            # plot_line(hand[0], hand[9])
            # plot_line(hand[2], hand[4])
            # plot_line(hand[5], hand[8])
            # if is_pointing_hand:
            #     plot_line(hand[5], hand[8], color='-r')
            # plot_line(hand[9], hand[12])
            # plot_line(hand[13], hand[16])
            # plt.text(p.x * w, p.y*h, str(i))
        bbox = self.find_bbox(self.palm_hand)
        plot_bbox(bbox)
        bbox = self.find_bbox(self.pointing_hand)
        plot_bbox(bbox)
        plt.show()
    
    def pred_using_mph(self, root_dir=None):
        if root_dir is not None:
            self.img_path = os.path.join(root_dir, self.img_path)
        img = cv2.imread(self.img_path)
        w, h, channel = img.shape


def read_dme_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        print('len data=',len(data), '(print from read_dme_json)')
    data_list = []
    for dat in data:
        data_list.append(DMEDataFromJson(dat))
    return data_list

def main():
    path1 = '../testing_json'
    path2 = '../training_json'
    path3 = '../validation_json'
    path = path3
    print(path)
    # read_dme_json(path1)
    data = read_dme_json(path2)
    # read_dme_json(path3)
    a = []
    for i in range(0,12):
        a.append(str(i))
    al = []
    for dat in data:
        if dat.gt in a and dat.gt not in al:
            al.append(dat.gt)
            # dat.plot()
            # print('cp',dat.img_path, os.path.join('.','temp', dat.img_name))


if __name__ =='__main__':
    main()