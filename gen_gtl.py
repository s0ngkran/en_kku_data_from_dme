import torch
import numpy as np
import pickle
import os
from torch.nn import functional as F
link = [[0,1] ,[0,3] ,[0,5] ,[0,7] ,[0,9], [1,2], [3,4], [5,6], [7,8], [9,10]]
link = [[0,1]]
link25 =  [[0,1],[1,2],[2,3],[0,4],[5,6],[6,7],[7,8],[4,9],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[21,22],[22,23],[23,24]]
link12from25 = [[0,1],[1,3],[0, 4],[4,5],[4,9],[4,13],[4,17],[5,8],[9,12],[13,16],[17,20],[21,24]] #len=12 (12*2=24)
index_point12_from25 = [0,1,3,4,5,8,9,12,13,16,17,20,21,24] #len=14
def gt_vec(width, height, point, size=20, link=link):
    ans = torch.zeros( len(link)*2, width, height)
    x, y = np.where(ans[0]==0)
    for index, (partA, partB) in enumerate(link):
        vec = point[partB]-point[partA]
        length = np.sqrt(vec[0]**2+vec[1]**2)
        u_vec = vec/length
        u_vec_p = np.array([u_vec[1], -u_vec[0]])

        tempx = x-point[partA][0]
        tempy = y-point[partA][1]
        temp_ = []
        temp_.append(tempx)
        temp_.append(tempy)
        temp = np.stack(temp_)
  
        c1 = np.dot(u_vec,temp)
        c1 = (0<=c1) & (c1<=length)
        c2 = abs(np.dot(u_vec_p,temp)) <= size
        condition = c1 & c2
       
        ans[ index*2] = torch.tensor(u_vec[0] * condition).reshape(width, height)  #x
        ans[ index*2+1] = torch.tensor(u_vec[1] * condition).reshape(width, height) #y
    return ans

def index_from_link(link):
    mydict = {}
    for a, b in link:
        mydict.update({
            str(a):1,
            str(b):1,
        })
    
    # convert to list
    index = []
    for key in mydict.keys():
        index.append(int(key))

    return index

def gt_vec_with_covered(width, height, keypoint, size=20, link=link12from25):
    '''
    keypoint = [[x,y,covered], [x,y,covered], ...]
    '''
    def gen_covered_link_map():
        print('warning! you have to check the link size 12 or 25')
        # create covered link map <covered_keypoint, covered_link>
        covered_link_map = {}
        for covered_link_index, (a, b) in enumerate(link):
            covered_link_index = str(covered_link_index)
            a, b = str(a), str(b)
            if a not in covered_link_map:
                covered_link_map.update({
                    a:covered_link_index,
                })
            else:
                covered_link_map[a] += ','+covered_link_index
            if b not in covered_link_map:
                covered_link_map.update({
                    b:covered_link_index,
                })
            else:
                covered_link_map[b] += ','+covered_link_index
        return covered_link_map

    def check_covered(covered_link_map, keypoint):
        '''
        [input]
        covered_link_map is a {} that is generated from gen_covered_link_map()
        keypoint = [[x,y,covered], [x,y,covered], ...]

        [output]
        - gtl
        - coverd_link
        '''
        # get covered_index
        covered_index = []
        covered_point = [] # len == 12
        link12 = index_from_link(link)
        for index, (x,y, covered) in enumerate(keypoint):
            if index in link12:
                covered = True if covered=='1' else False
                covered_point.append(covered)
                if covered:
                    covered_index.append(index)
                    
        # gen covered link
        covered_link = {}
        for index in covered_index:
            index = str(index)
            update_list = covered_link_map[index].split(',')
            for ind in update_list:
                covered_link.update({
                    ind:'covered',
                })

        ans = [False for i in range(len(link)*2)]   
        for i in covered_link.keys():
            ans[int(i)*2] = True
            ans[int(i)*2+1] = True

        covered_link = ans
        return covered_link


    # gen covered link
    covered_link_map = gen_covered_link_map()
    covered_link = check_covered(covered_link_map, keypoint)

    # manage keypoint to point (old version)
    point = []
    for x,y,covered in keypoint:
        point.append(np.array([x, y]).astype(np.float32))

    # gen gtl (old version)
    ans = torch.zeros( len(link)*2, width, height )
    x, y = np.where( ans[0]==0 )

    # gtl_mask
    gtl_mask = torch.zeros(ans.shape)
    
    for index, (partA, partB) in enumerate(link):
        vec = point[partB]-point[partA]
        length = np.sqrt(vec[0]**2+vec[1]**2)
        u_vec = vec/length
        u_vec_p = np.array([u_vec[1], -u_vec[0]])

        tempx = x-point[partA][0]
        tempy = y-point[partA][1]
        temp_ = []
        temp_.append(tempx)
        temp_.append(tempy)
        temp = np.stack(temp_)
  
        c1 = np.dot(u_vec,temp)
        c1 = (0<=c1) & (c1<=length)
        c2 = abs(np.dot(u_vec_p,temp)) <= size
        mask = c1 & c2
        
        ans[ index*2] = torch.tensor(u_vec[0] * mask).reshape(width, height)  #x
        ans[ index*2+1] = torch.tensor(u_vec[1] * mask).reshape(width, height) #y

        gtl_mask[index*2][mask] = 1 # use float instead of bool for interpolate in the next process
        gtl_mask[index*2+1][mask] = 1

        # manage special case of covered link
        # azi 0 and azi 90
        case1 = u_vec[0] == 0 and u_vec[1] != 0
        case2 = u_vec[0] != 0 and u_vec[1] == 0
        if case1 or case2:
            covered_link[index] = False

    gtl = ans
    return gtl, gtl_mask, covered_link
def distance (p1, p2):
    distx = (p1[0]-p2[0])**2
    disty = (p1[1]-p2[1])**2
    return (distx+disty)**0.5
def gen_12_keypoint_with_covered_link(keypoint, width, height, sigma=0.4):
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    small_sigma = dist_finger * sigma
    
    keypoint = [np.array(i) for i in keypoint]
    gtl, gtl_mask, covered_link = gt_vec_with_covered(width, height, keypoint, size=small_sigma, link=link12from25)
    
    gtl = F.interpolate(gtl.unsqueeze(0), size=(45,45), mode='nearest').squeeze(0)

    # manage gtl_mask
    gtl_mask = F.interpolate(gtl.unsqueeze(0), size=(45,45), mode='bicubic').squeeze(0)
    gtl_mask[gtl_mask > 0.5] = 1
    gtl_mask[gtl_mask <= 0.5] = 0
    gtl_mask = gtl_mask.type(torch.BoolTensor)

    return gtl, gtl_mask, covered_link

def gen_25_keypoint(keypoint, width, height):
    link25 =  [[0,1],[1,2],[2,3],[0,4],[5,6],[6,7],[7,8],[4,9],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[21,22],[22,23],[23,24]]
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    small_sigma = dist_finger*0.4
    
    keypoint = [np.array(i) for i in keypoint]
    gtl = gt_vec(width, height, keypoint,size=small_sigma, link=link25)
    
    gtl = F.interpolate(gtl.unsqueeze(0), size=(45,45), mode='nearest').squeeze(0)
    return gtl
    
def test_gt_vec():
    import matplotlib.pyplot as plt 
    link25 =  [[0,1],[1,2],[2,3],[0,4],[5,6],[6,7],[7,8],[4,9],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[21,22],[22,23],[23,24]]
    i = 1
    
    with open('pkl480.pkl', 'rb') as f:
        data = pickle.load(f)
    dat = data[str(i)]
    point = dat['keypoint']
    keypoint = point
    
    finger_link = [[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
    dist_finger = [distance(keypoint[i], keypoint[j]) for i,j in finger_link]
    dist_finger = sum(dist_finger)/len(dist_finger)
    
    small_sigma = dist_finger*0.4
    
    point = [np.array(i) for i in point]
    gtl = gt_vec(480,480, point,size=small_sigma, link=link25)
    
    
    gtl = F.interpolate(gtl.unsqueeze(0), size=(120,120), mode='nearest').squeeze(0)
    print(gtl.shape)
    gtl = gtl.mean(0)
    
    thres = 120/480 
    for x,y in point:
        x, y = x*thres, y*thres
        plt.plot(x,y,'r.')
        
    plt.imshow(gtl.transpose(0,1))
    plt.show()
def gen_gtl_folder(gt_file, savefolder, dim1, dim2, size):
    assert gt_file[-6:] == '.torch'
    assert savefolder[-1] == '/'
    from torch.nn import functional as F

    gt = torch.load(gt_file)
    keypoint = gt['keypoint']
    for i in range(len(keypoint)):
        if i == 0: continue
        
        # if i<=350: continue
        point = [np.array(x) for x in keypoint[i]]
        width, height = dim1
        gtl = gt_vec(width,height,point,size=size)
        
        gtl = F.interpolate(gtl.unsqueeze(0), dim2, mode='nearest').squeeze()
        # import matplotlib.pyplot as plt
        # plt.imshow(gtl[0])
        # plt.show()
        name = str(i).zfill(10)
        torch.save(gtl, savefolder+name)
        print(name, i, len(keypoint))
    print('fin all')
def test_gtl():
    import matplotlib.pyplot as plt 
    import cv2
    from torch.nn import functional as F
    img = cv2.imread('example_folder/0000000010.bmp') # y,x,ch
    print(img.shape)
    img = torch.FloatTensor(img/255).transpose(0,2) # ch,x,y

    gtl = torch.load('example_folder_gtl/0000000010')
    
    gtl = F.interpolate(gtl.unsqueeze(0), (360,360), mode='nearest').squeeze()
    # gtl = gtl.mean(0)
    # ans = img[0]*0.01+ gtl*0.5
    ans = gtl[0]
    plt.imshow(ans)
    plt.colorbar()
    plt.show()
def ex_gen_gtl_folder():
    gt_folder = 'testing/pkl/'
    savefolder = 'testing/gtl/'
    dim1 = (360,360)
    dim2 = (45,45)
    gen_gtl_folder(gt_folder, savefolder, dim1, dim2)
def test_gt_vec_with_covered():
    width = 45
    height = 45

if __name__ == "__main__":
    # lst = ['gt_random_background_aug.torch','gt_replaced_background.torch','gt_replaced_green.torch']
    # lst2 = ['random_background','replaced_background','replaced_green']
    # a = zip(lst, lst2)
    # for gt, savefol in a:
    #     gt_file = 'training/'+gt
    #     savefolder = 'training/gtl/'+savefol+'/'
    #     dim1 = (360,360)
    #     dim2 = (45,45)
    #     size = 10
    #     gen_gtl_folder(gt_file, savefolder, dim1, dim2, size)
    #     print('-----------------------------')
    pass
