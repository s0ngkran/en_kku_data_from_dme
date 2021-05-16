import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy
import os
from shutil import copyfile
from do_augment import *
import torch
from gen_gts import gen_12_keypoint_with_covered_point as gen_gts
from gen_gtl import gen_12_keypoint_with_covered_link as gen_gtl

# open data
with open('hands_json_with_name', 'r') as f:
    data = json.load(f)

def create_augment(dat):
    # manage img and keypoint
    img = cv2.imread('hand/'+dat['path'])
    dim = (360, 360)
    w, h = dim
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    keypoint = dat['keypoint']
    ans_keypoint = []
    only_covered = []
    for x, y, covered in keypoint:
        x = x*w
        y = y*h
        ans_keypoint.append([x,y])
        only_covered.append(covered)
    
    keypoint = ans_keypoint

    # do augment
    if dat['hand_side'] == 'L':
        img, keypoint = aug_flip(img, keypoint)
        print('from L to R')
    # img, keypoint = aug_scale_and_shift(img, keypoint)
    # img, keypoint = aug_rotate(img, keypoint)
    # img, keypoint = aug_noise(img, keypoint)

    # add covered
    keypoint_with_covered = []
    for keypoint_, only_covered_ in zip(keypoint, only_covered):
        x, y = keypoint_
        covered = only_covered_
        keypoint_with_covered.append([x,y,covered])
    return img, keypoint_with_covered

def create_gt(keypoint):
    w, h = 360, 360
    # gen gtl
    gtl, covered_link = gen_gtl(keypoint, w, h)
    # gen gts
    gts, covered_point = gen_gts(keypoint, w, h)
    return gts, gtl, covered_point, covered_link

# validation json
validation_json = []

# read
for cnt, dat in enumerate(data):
    print(cnt, len(data))
    user = dat['user']
    if user not in ['Jammy']:
        continue

    ori_keypoint = dat['keypoint']
    img_path = dat['path']
    # collect coverd
    covered = []
    for _, __, c in ori_keypoint:
        covered.append(c) 
    
    replica = 1
    for i in range(replica):
        new_dat = dat.copy()
        # create augment
        img, keypoint = create_augment(new_dat)
        new_path = 'validation_set/'+str(i)+img_path
        
        # save aug_img and aug_keypoint
        cv2.imwrite(new_path, img)

        # create gt
        gts, gtl, covered_point, covered_link = create_gt(keypoint)
        gts_path = 'validation_set/'+str(i)+img_path+'.gts'
        gtl_path = 'validation_set/'+str(i)+img_path+'.gtl'
        torch.save(gts, gts_path)
        torch.save(gtl, gtl_path)
    
        # add covered value to keypoint
        new_keypoint = []
        for keypoint_, c in zip(keypoint, covered):
            keypoint_.append(c)
            new_keypoint.append(keypoint_)
        new_dat['keypoint'] = copy.copy(new_keypoint)
        new_dat['status'] = 'validation'
        new_dat['path'] = copy.copy(new_path)
        new_dat['gts'] = copy.copy(gts_path)
        new_dat['gtl'] = copy.copy(gtl_path)
        new_dat['covered_point'] = copy.copy(covered_point)
        new_dat['covered_link'] = copy.copy(covered_link)

        validation_json.append(new_dat)
    
# write validation_json
with open('validation_json', 'w') as f:
    json.dump(validation_json, f)
    
