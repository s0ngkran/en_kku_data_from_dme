import torch
import torch.nn.functional as F
import sys
import copy
sys.path.append('..')
from util.Logger import Logger
logger = Logger('log_loss')
def loss_func(out, gt_l, gt_s, covered_point, covered_link):  #, covered):
    assert len(out) == 4, 'check your output stage' # L1, L2, S1, S2
    batch = gt_l.shape[0]
    width = gt_l.shape[2]
    height = gt_l.shape[3]
    shape = gt_l.shape     # batch, depth, width, height
    loss = 0
    loss_gts = 0
    loss_gtl = 0
    
    ############################  gts  ##########################################
    thres = 0.01
    thres_zero = 4/96 *0.6
    thres_non  = 0.4
    # torch.ones(gt_s.shape, dtype=torch.float32).cuda()
    
    for i in [2, 3]:
        pred_s = out[i].cpu()
        # weight[gt_s < thres] *= thres_zero
        # weight[gt_s >= thres] *= thres_non
        loss_ = ((pred_s - gt_s)**2)
        loss_[gt_s < thres] *= thres_zero
        loss_[gt_s >= thres] *= thres_non

        # manage coverd_point
        # for batch_ in range(batch):
        #     for depth_ in range(gt_s.shape[1]):
        #         if covered_point[batch_, depth_]: # covered == True
        #             loss_[batch_, depth_] = 0
        loss_[covered_point == True] = 0
        
        # loss of this Stage 
        # loss_ /= width   # avoid +inf value    

        loss_gts += torch.sum(loss_)
        loss += loss_gts 

    ######################### gtl ##########################################
    
    # weight - class balancing
    thres_zero = 1/99 *0.6
    thres_non  = 0.4
    for i in [0, 1]:
        pred_l = out[i].cpu()
        # weight[gt_l==0] *= thres_zero
        # weight[gt_l!=0] *= thres_non
        loss_ = (pred_l - gt_l)**2
        loss_[gt_l==0] *= thres_zero
        loss_[gt_l!=0] *= thres_non
       
        
        # old version
        # # manage coverd_point
        # for batch_ in range(batch):
        #     for depth_ in range(gt_l.shape[1]):
        #         if covered_link[batch_, depth_]: # covered == True
        #             loss_[batch_, depth_] = 0
        
        # if (torch.isinf(loss_)).any() : 
        #     print(4, loss_)
        #     1/0
        
        
        # new version
        loss_[covered_link == True] = 0 
        
        
        # loss of this Stage   
        # loss_ /= width   # avoid +inf value  
        loss_gtl = torch.sum(loss_)
        loss += loss_gtl
        
    # print('---',loss)
    loss /= batch
    loss_gts /= batch
    loss_gtl /= batch
    return loss, loss_gts, loss_gtl
def testloss():
    import copy
    torch.manual_seed(7)
    gts = torch.randn([5,6,120,120]).type(torch.float16).cuda()
    gtl = torch.randn([5,46,120,120]).type(torch.float16).cuda()
    
    pred_l1 = torch.randn([5,46,120,120]).type(torch.float16).cuda()
    pred_l2 = gtl.clone()
    pred_s1 = gts.clone()
    pred_s2 = gts.clone()
    
    covered_point = []
    covered_point_ = torch.tensor([False for i in range(6)], dtype=torch.bool)
    # covered_point_[2] = True
    for i in range(5):
        covered_point.append(covered_point_)
    covered_point = torch.stack(covered_point)
    cov_p = covered_point.clone()
    
    covered_point = []
    covered_point_ = torch.tensor([False for i in range(46)], dtype=torch.bool)
    # covered_point_[2] = True
    for i in range(5):
        covered_point.append(covered_point_)
    covered_point = torch.stack(covered_point)
    cov_l = covered_point.clone()
    
    
    out = (pred_l1, pred_l2, pred_s1, pred_s2)
    loss = loss_func(out, gtl, gts, cov_p, cov_l)
    print(loss)
def test_size(): # by rand
    out = [torch.rand(2,2,60,60) for i in range(4)]
    gts = torch.rand(2,2,60,60)
    gtl = torch.rand(2,2,60,60)
    print('input len=', len(out), out[0].shape)
    loss = loss_func(out, gtl, gts)
    print('loss=', loss)

if __name__ == '__main__':
    # testloss()
    a = torch.rand(2,3)
    b = a.clone().fill_(0)
    print(b)
