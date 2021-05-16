from torch.nn import functional as F

import torch
import os
index_point12_from25 = [0,1,3,4,5,8, 9,12,13,16,17,20, 21, 24]#len=12
def gen_gts(cx,cy,width,height,sigma):
    # cx -> center x
    # cy -> center y
    # width, height of image
    # sigma -> size of heat
    emt = torch.zeros([width,height])
    x, y = torch.where(emt==0)
    distx = (x-cx).float()
    disty = (y-cy).float()
    dist = distx**2+disty**2
    ans = torch.exp(-(dist/sigma**2).float())
    ans = ans.reshape([width,height])
    
    # example code
    # import torch
    # import matplotlib.pyplot as plt
    # a = gen_gts(200,200,400,600,10)
    # plt.imshow(a)
    # plt.colorbar()
    # plt.show()
    return ans
def distance (p1, p2):
    distx = (p1[0]-p2[0])**2
    disty = (p1[1]-p2[1])**2
    return (distx+disty)**0.5

def gen_25_keypoint(keypoint, width, height):
    '''
    finger_link is used for calculate the ratio of hand-scale
    '''
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    big_link = [[4,5],[4,9],[4,13],[4,17],[4,0]]
    dist_big = [distance(keypoint[i], keypoint[j]) for i,j in big_link]
    dist_big = sum(dist_big)/len(dist_big)
    
    small_sigma = dist_finger*0.4
    big_sigma = dist_big*0.6
    gts = []
    
    ### config ####
    for ind, (x,y) in enumerate(keypoint):
        if ind in [0,1,4,21]:
            sigma = big_sigma
        else:
            sigma = small_sigma
        gts.append(gen_gts(x,y,width, height, sigma))
    gts = torch.stack(gts)
    gts = F.interpolate(gts.unsqueeze(0), size=(45,45), mode='bicubic').squeeze(0)
    return gts


def gen_12_keypoint_with_covered_point(keypoint, width, height, sigma=0.4):
    '''
    finger_link is used for calculate the ratio of hand-scale
    '''
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    big_link = [[4,5],[4,9],[4,13],[4,17],[4,0]]
    dist_big = [distance(keypoint[i], keypoint[j]) for i,j in big_link]
    dist_big = sum(dist_big)/len(dist_big)
    
    small_sigma = dist_finger * sigma
    big_sigma = dist_big * sigma * 1.3
    gts = []
    
    ### config ####
    covered_point = []
    for ind, (x,y, covered) in enumerate(keypoint):
        if ind in index_point12_from25:
            covered_point.append(covered)
            if ind in [0,1,4,21]:
                sigma = big_sigma
            else:
                sigma = small_sigma
            gts.append(gen_gts(x,y,width, height, sigma))
    gts = torch.stack(gts)
    gts = F.interpolate(gts.unsqueeze(0), size=(45,45), mode='bicubic').squeeze(0)

    # gen gts mask
    gts_mask = torch.zeros(gts.shape)
    gts_mask[gts > 0.01] = 1
    gts_mask = gts_mask.type(torch.bool)

    return gts, gts_mask, covered_point