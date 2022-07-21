import json
from collections import Counter
from utils import is_point_in_rotated_box
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys
from iou_cal import BBoxIOU, RotatedRect, get_iou
sys.path.insert(0, '../DME_VGG_POH/')
sys.path.insert(0, '../../mediaPipe_hand/apply_to_TFS/')
sys.path.insert(0, '../../mediaPipe_hand/MediaPipePyTorch/')
from all_results_MPH import MPH, MPHResult
from read_json import DMEDataFromJson, read_dme_json


class Point:
    def __init__(self, data=None, x=None, y=None):
        if x is None or y is None:
            assert type(data) == list
            assert len(data) == 3
            self.x, self.y, self.z = data
        else:
            self.x, self.y = x, y

class PointMPH:
    def __init__(self, data=None, x=None, y=None):
        if x == None and y == None:
            self.x = data['X']
            self.y = data['Y']
            self.z = data['Z']
        else:
            self.x = x
            self.y = y
        self.dist = None
        self.tag = None
        self.index = None

    def distance(self, p, w=None, h=None):
        dist = ((p.x - self.x)*w)**2 + ((p.y - self.y)*h)**2
        dist = dist**0.5
        return dist

class MPHClassification:
    def __init__(self, handedness):
        assert type(handedness) == str
        x, label = handedness.split('\n}')[0].split('\n  label: ')
        label = label.strip('"')
        x, score = x.split('\n  score: ')
        score = float(score)
        index = x[-1]

        self.label = label
        self.score = score
        self.hand_index = index

    def __str__(self):
        return self.hand_index + ' ' + self.label


class MPHObject:
    def __init__(self, data):
        self.img_path = data['img_path']
        self.handedness = data['handedness']
        self.hand_landmarks = data['hand_landmarks']
        self.gt = None

        def get_classification():
            classification = []
            for i in self.handedness:
                classification.append(MPHClassification(i))
            classification.sort(key=lambda x: x.hand_index)
            return classification

        self.classification = get_classification()

        def read_hands(data):
            hands = []
            for hand_ in data:
                hand = []
                for point in hand_:
                    hand.append(PointMPH(point))
                hands.append(hand)
            if hands == []:
                print('empty hand')
            return hands
        self.hands = read_hands(self.hand_landmarks)
        self.n_hand = len(self.hands)
        self.point_k = None
        self.palm_hand = None
        self.pointing_hand = None

        # const
        self.finger_index = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17]
        self.finger_label = ['A', 'B', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'J']

        for hand in self.hands:
            p0 = hand[0]
            p9 = hand[9]
            x_mean = (p0.x + p9.x)/2
            y_mean = (p0.y + p9.y)/2
            is_pointing = self.is_pointing_hand(hand)
            if is_pointing:
                self.pointing_hand = hand
            else:  # this is palm hand
                self.palm_hand = hand
                self.point_k = PointMPH(x=x_mean, y=y_mean)

    def is_pointing_hand(self, hand):
        def vector_dir(p_start, p_end):
            distance = [p_end.x - p_start.x, p_end.y - p_start.y]
            norm = (distance[0] ** 2 + distance[1] ** 2)**0.5
            direction = [distance[0] / norm, distance[1] / norm]
            # example
            # d = vector_dir(Point(x=0,y=0), Point(x=10,y=0))
            return direction

        def angle_between(v0, v1):
            angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            angle = np.degrees(angle)
            angle = abs(angle)
            return angle

        index_finger = vector_dir(hand[5], hand[8])  # index_finger
        middle_finger = vector_dir(hand[9], hand[12])  # middle_finger
        ring_finger = vector_dir(hand[13], hand[16])
        pink_finger = vector_dir(hand[17], hand[20])

        a = angle_between(index_finger, middle_finger)
        b = angle_between(index_finger, ring_finger)
        c = angle_between(index_finger, pink_finger)
        thres = 90
        # thres = np.pi/2
        if a > thres and b > thres and c > thres:
            return True
        return False

    def plot(self, root_dir=None):
        if root_dir is not None:
            self.img_path = os.path.join(root_dir, self.img_path)
        img = cv2.imread(self.img_path)
        w, h, channel = img.shape

        def plot_line(p1, p2, w=w, h=h, color='-y', is_scaled=False):
            if is_scaled:
                w = 1
                h = 1
            x1, y1 = p1.x * w, p1.y*h
            x2, y2 = p2.x * w, p2.y*h
            plt.plot((x1, x2), (y1, y2), color)
        img = cv2.flip(img, 1)
        plt.imshow(img[:, :, (2, 1, 0)])
        text = ''
        for i in self.classification:
            text += str(i) + '\n'
        text += '[0]yellow  [1]pink'
        plt.title(text)
        for hand_index, hand in enumerate(self.hands):
            is_pointing_hand = self.is_pointing_hand(hand)
            for i, p in enumerate(hand):
                if i in self.finger_index:
                    if i in [8, 5] and is_pointing_hand:  # index finger
                        color = 'or'
                    else:
                        color = 'ob'
                    plt.plot(p.x * w, p.y*h, color)
            if hand_index == 0:
                color = '-y'
            else:
                color = '-m'
            plot_line(hand[0], hand[2], color=color)
            plot_line(hand[0], hand[9], color=color)
            plot_line(hand[2], hand[4], color=color)
            plot_line(hand[5], hand[8], color=color)
            plot_line(hand[9], hand[12], color=color)
            plot_line(hand[13], hand[16], color=color)
            if is_pointing_hand:
                plot_line(hand[5], hand[8], color='-r')
            # plt.text(p.x * w, p.y*h, str(i))
        k = self.point_k
        plt.plot(k.x*w, k.y*h, 'ob')
        # self.pred_nearest()
        plt.show()

    def add_gt(self, gt_list):
        assert type(gt_list) == list
        assert type(gt_list[0]) == DMEDataFromJson
        # find gt that has the same img_path
        for gt in gt_list:
            if gt.img_path == self.img_path:
                self.gt = gt
                return
        assert False, 'cannot add gt to this pred; no same img_path'

    def get_result_bbox(self):
        '''
        1. check bbox -> iou
            gt has 2 hands -> R, L
            pred 
                - 1 hand == false
                - 2 hands and false R, L == false
                - 2 hands and correct R, L == true


        '''
        # get gt
        self.gt


def read_mph_result(path, gt_list):
    with open(path, 'r') as f:
        data = json.load(f)
    pred_list = []
    for dat in data:
        _ = MPHObject(dat)
        _.add_gt(gt_list)
        pred_list.append(_)
    return pred_list


def get_pred_from_original_MPH_model(gt_list):
    # pred
    path_mph_result = '../result_from_mph_dme_testing_set.json'
    pred_list = read_mph_result(path_mph_result, gt_list)
    p = pred_list[0]
    print(len(pred_list))
    pred_list[0].plot('../')

def get_min_max_avg(alist):
    if alist is not None:
        alist = alist.split('|')[:-1]
        alist = [float(i) for i in alist]
        alist.sort(key=lambda x: x)
        mn = alist[0]
        mx = alist[-1]
        assert mx >= mn
        avg = sum(alist)/len(alist)
        return mn, mx, avg
    else:
        return None, None, None

def evaluate():
    print('start...')
    # ground truth
    path = '../testing_json'
    gt_list = read_dme_json(path)

    ############
    gt = DMEDataFromJson
    pred = MPHResult
    mph = MPH
    ################
    # what i need
    can_detect_both_hands = []  # problem with is_pointing_hand() ->
    can_detect_pointing_hand = []
    can_detect_palm_hand = []
    iou_pointing = []
    iou_palm = []
    v_ref_palm = []
    v_ref_pointing = []
    v_center_to_finger_pointing = []
    v_center_to_finger_palm = []
    v_finger_pointing = []
    v_finger_palm = []
    expanded_result_pointing = []
    expanded_result_palm = []
    flag_pointing = []
    flag_palm = []
    handed_pointing = []
    handed_palm = []
    landmark_pointing = []
    landmark_palm = []
    landmark_pointing_min = []
    landmark_pointing_max = []
    landmark_pointing_avg = []
    landmark_palm_min = []
    landmark_palm_max = []
    landmark_palm_avg = []
    mph = MPH(root_dir='../../mediaPipe_hand/MediaPipePyTorch/')
    c = []
    for i, gt in enumerate(gt_list):
        # if i >= 10: break
        # if iii != 28: continue

        # gt = gt_list[2]
        path = gt.img_path
        img = cv2.imread(os.path.join('..', path))
        # pred = mph.pred(img=img, draw=False, raw=False, gt=gt)
        pred = mph.pred(img=img, draw=True, raw=False)
        pred.plot(img=img)
        plt.title(str(i))

        # reading
        plt.show()
        continue

        # To draw IOU, I need to get the image.
        # then, return the rotated rectangle.


        #######
        #######
        #######
        #######

        # debug

        #######
        #######
        #######
        #######
        #######
        #######
        # continue

        # plt.imshow(pred)
        # plt.show()
        # 1/0

        # pred.plot(img=img)
        # plt.show()
        # 1/0

        ############
        _gt = DMEDataFromJson
        _pred = MPHResult
        _mph = MPH
        ################

        can_detect_pointing_hand.append(pred.can_detect_pointing_hand)
        can_detect_palm_hand.append(pred.can_detect_palm_hand)

        v_ref_pointing.append(pred.vec_ref_pointing)
        v_ref_palm.append(pred.vec_ref_palm)

        v_center_to_finger_pointing.append(pred.v_center_to_finger_pointing)
        v_center_to_finger_palm.append(pred.v_center_to_finger_palm)

        v_finger_pointing.append(pred.v_finger_pointing)
        v_finger_palm.append(pred.v_finger_palm)

        expanded_result_pointing.append(pred.expanded_result_pointing)
        expanded_result_palm.append(pred.expanded_result_palm)

        flag_pointing.append(pred.flag_pointing)
        flag_palm.append(pred.flag_palm)

        handed_pointing.append(pred.handed_pointing)
        handed_palm.append(pred.handed_palm)

        landmark_pointing.append(pred.landmark_pointing)
        landmark_palm.append(pred.landmark_palm)

        ### stat
        mn, mx, avg = get_min_max_avg(pred.landmark_pointing)
        landmark_pointing_min.append(mn)
        landmark_pointing_max.append(mx)
        landmark_pointing_avg.append(avg)

        mn, mx, avg = get_min_max_avg(pred.landmark_palm)
        landmark_palm_min.append(mn)
        landmark_palm_max.append(mx)
        landmark_palm_avg.append(avg)


        # fix these lines
        # def get_keypoint_diff():
        #     # for i, hand in enumerate(pred.hands):

        # hands = fill_none(pred.hands)
        # landmark0.append(hands[0])
        # landmark1.append(hands[1])
        #####################################

        # _v_ref = pred.

        # for p in pred.point_ref:
        #     p.plot('og')

        # plt.show()
        # 1/0

        # for i in pred.bboxs:
        #     print(i)
        # 1/0

        # def get_iou_bbox():
        #     if len(pred.bboxs) < 2:
        #         print('bbox < 2', len(pred.bboxs))
        #         return None

        #     # gt
        #     d1 = gt.find_bbox(gt.palm_hand)
        #     d2 = gt.find_bbox(gt.pointing_hand)
        #     gt_bbox1 = BBoxIOU(d1)
        #     gt_bbox2 = BBoxIOU(d2)

        #     # pred
        #     pred_bbox1, pred_bbox2 = pred.bboxs
        #     pred_bbox1 = BBoxIOU(pred_bbox1.tuple)
        #     pred_bbox2 = BBoxIOU(pred_bbox2.tuple)

        #     ######## plot

        #     w, h, channel = img.shape
        #     def plot_line(p1, p2, w=w, h=h, color='-y', is_scaled=False):
        #         if is_scaled:
        #             w = 1
        #             h = 1
        #         x1, y1 = p1.x * w, p1.y*h
        #         x2, y2 = p2.x * w, p2.y*h
        #         plt.plot((x1, x2), (y1, y2), color)
        #     # img = cv2.flip(img, 1)
        #     def plot_bbox(bbox, color='-y'):
        #         min_x, max_x, min_y, max_y = bbox
        #         plot_line(Point(x=min_x, y=min_y), Point(x=max_x, y=min_y), is_scaled=True)
        #         plot_line(Point(x=min_x, y=max_y), Point(x=max_x, y=max_y), is_scaled=True)
        #         plot_line(Point(x=min_x, y=min_y), Point(x=min_x, y=max_y), is_scaled=True)
        #         plot_line(Point(x=max_x, y=min_y), Point(x=max_x, y=max_y), is_scaled=True)

        #     # plot
        #     plt.imshow(img[:,:,(2,1,0)])
        #     plot_bbox(gt_bbox1.for_iou(), '-g')
        #     plot_bbox(gt_bbox2.for_iou(), '-g')
        #     plot_bbox(pred_bbox1.for_iou(), '-m')
        #     plot_bbox(pred_bbox2.for_iou(), '-m')
        #     plt.show()
        #     1/0

        #     def find_max_iou_of_matching(gt_bbox1, gt_bbox2, pred_bbox1, pred_bbox2):
        #         iou1 = get_iou(gt_bbox1, pred_bbox1) + get_iou(gt_bbox2, pred_bbox2)
        #         iou2 = get_iou(gt_bbox1, pred_bbox2) + get_iou(gt_bbox2, pred_bbox1)
        #         if iou1 > iou2:
        #             return iou1
        #         else:
        #             return iou2
        #     iou = find_max_iou_of_matching(gt_bbox1, gt_bbox2, pred_bbox1, pred_bbox2)
        #     return iou

        # iou = get_iou_bbox()
        # print(iou)
        # pred.plot(img)

        # plt.show()

    sep = '\t'
    text = 'can_detect_pointing_hand\t'
    text += sep.join([str(x) for x in can_detect_pointing_hand]) + '\n'
    text += 'can_detect_palm_hand\t'
    text += sep.join([str(x) for x in can_detect_palm_hand]) + '\n'
    text += 'v_ref_pointing\t'
    text += sep.join([str(x) for x in v_ref_pointing]) + '\n'
    text += 'v_ref_palm\t'
    text += sep.join([str(x) for x in v_ref_palm]) + '\n'
    text += 'v_center_to_finger_pointing\t'
    text += sep.join([str(x) for x in v_center_to_finger_pointing]) + '\n'
    text += 'v_center_to_finger_palm\t'
    text += sep.join([str(x) for x in v_center_to_finger_palm]) + '\n'
    text += 'v_finger_pointing\t'
    text += sep.join([str(x) for x in v_finger_pointing]) + '\n'
    text += 'v_finger_palm\t'
    text += sep.join([str(x) for x in v_finger_palm]) + '\n'
    text += 'expanded_result_pointing\t'
    text += sep.join([str(x) for x in expanded_result_pointing]) + '\n'
    text += 'expanded_result_palm\t'
    text += sep.join([str(x) for x in expanded_result_palm]) + '\n'
    text += 'flag_pointing\t'
    text += sep.join([str(x) for x in flag_pointing]) + '\n'
    text += 'flag_palm\t'
    text += sep.join([str(x) for x in flag_palm]) + '\n'
    text += 'handedness_pointing\t'
    text += sep.join([str(x) for x in handed_pointing]) + '\n'
    text += 'handedness_palm\t'
    text += sep.join([str(x) for x in handed_palm]) + '\n'
    text += 'distance_landmark_pointing\t'
    text += sep.join([str(x) for x in landmark_pointing]) + '\n'
    text += 'distance_landmark_palm\t'
    text += sep.join([str(x) for x in landmark_palm]) + '\n'
    text += 'distance_landmark_pointing_min\t'
    text += sep.join([str(x) for x in landmark_pointing_min]) + '\n'
    text += 'distance_landmark_pointing_max\t'
    text += sep.join([str(x) for x in landmark_pointing_max]) + '\n'
    text += 'distance_landmark_pointing_avg\t'
    text += sep.join([str(x) for x in landmark_pointing_avg]) + '\n'
    text += 'distance_landmark_palm_min\t'
    text += sep.join([str(x) for x in landmark_palm_min]) + '\n'
    text += 'distance_landmark_palm_max\t'
    text += sep.join([str(x) for x in landmark_palm_max]) + '\n'
    text += 'distance_landmark_palm_avg\t'
    text += sep.join([str(x) for x in landmark_palm_avg]) + '\n'

    text.replace(',', ';')
    text.replace('\t', ',')

    filename = './mph_study.csv'
    with open(filename, 'w') as f:
        # f.write(text)
        print('writed', filename)

    # plt.show()

    '''
    Intermediate Results
    1. check bbox -> iou
        gt has 2 hands -> R, L
        pred 
            - 1 hand == false
            - 2 hands and false R, L == false
            - 2 hands and correct R, L == true

    2. check correction using hand scale
    '''

    # for p in dat.hand:
    #     print(p.x, p.y)
    # dat.plot('../')


def main():
    # is_point_outside_rotated_box()
    evaluate()


if __name__ == '__main__':
    main()
