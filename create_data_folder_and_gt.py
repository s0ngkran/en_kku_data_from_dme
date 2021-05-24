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
    img, keypoint = aug_scale_and_shift(img, keypoint)
    img, keypoint = aug_rotate(img, keypoint)
    img, keypoint = aug_noise(img, keypoint)

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

def create_curriculum_gt(keypoint):
    w, h = 360, 360
    
    gts, gts_mask, gts_covered, gtl, gtl_mask, gtl_covered = [],[],[],[],[],[]
    for sigma in [0.7, 0.55, 0.4]:
        # gen gtl
        gtl_, gtl_mask_, gtl_covered_ = gen_gtl(keypoint, w, h, sigma=sigma)
        gtl.append(gtl_)
        gtl_mask.append(gtl_mask_)
        gtl_covered.append(gtl_covered_)

        # gen gts
        gts_, gts_mask_, gts_covered_ = gen_gts(keypoint, w, h, sigma=sigma)
        gts.append(gts_)
        gts_mask.append(gts_mask_)
        gts_covered.append(gts_covered_)
    
    gts = torch.stack(gts)
    gts_mask = torch.stack(gts_mask)
    gtl = torch.stack(gtl)
    gtl_mask = torch.stack(gtl_mask)

    return gts, gts_mask, gts_covered, gtl, gtl_mask, gtl_covered

# json
json_data = []

# write json_data     ############################# config
json_name = 'training'
# json_name = 'validation'
# json_name = 'testing'
augment_replica = 4

# read
save_folder = json_name + '_set_curriculum/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

for cnt, dat in enumerate(data):
    print(cnt, len(data))
    user = dat['user']
    if json_name == 'training':
        if user in ['DMW', 'Jammy']:   # for training_set
            continue
    elif json_name == 'testing':
        if user not in ['DMW']: # for testing_set
            continue
    elif json_name == 'validation':
        if user not in ['Jammy']: # for validation_set
            continue

    ori_keypoint = dat['keypoint']
    img_path = dat['path']
    # collect coverd
    covered = []
    for _, __, c in ori_keypoint:
        covered.append(c) 
    
    for i in range(augment_replica):
        new_dat = dat.copy()
        # create augment
        img, keypoint = create_augment(new_dat)
        new_path = save_folder + str(i)+img_path
        
        # save aug_img and aug_keypoint
        cv2.imwrite(new_path, img)

        # create gt
        gts, gts_mask, covered_point, gtl, gtl_mask, covered_link = create_curriculum_gt(keypoint)

        gts_path = save_folder + str(i)+img_path+'.gts'
        gts_mask_path = save_folder + str(i)+img_path+'.gts_mask'
        gtl_path = save_folder + str(i)+img_path+'.gtl'
        gtl_mask_path = save_folder + str(i)+img_path+'.gtl_mask'
        
        torch.save(gts, gts_path)
        torch.save(gtl, gtl_path)
        torch.save(gts_mask, gts_mask_path)
        torch.save(gtl_mask, gtl_mask_path)
    
        # add covered value to keypoint
        new_keypoint = []
        for keypoint_, c in zip(keypoint, covered):
            keypoint_.append(c)
            new_keypoint.append(keypoint_)
        new_dat['keypoint'] = copy.copy(new_keypoint)
        new_dat['status'] = json_name
        new_dat['path'] = copy.copy(new_path)
        new_dat['gts'] = copy.copy(gts_path)
        new_dat['gtl'] = copy.copy(gtl_path)
        new_dat['gts_mask'] = copy.copy(gts_mask_path)
        new_dat['gtl_mask'] = copy.copy(gtl_mask_path)
        new_dat['covered_point'] = copy.copy(covered_point)
        new_dat['covered_link'] = copy.copy(covered_link)

        json_data.append(new_dat)
        
with open(json_name + '.curriculum.json', 'w') as f:
    json.dump(json_data, f)
    
