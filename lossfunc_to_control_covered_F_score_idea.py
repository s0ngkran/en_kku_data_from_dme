import torch
import torch.nn.functional as F
import sys
import copy
sys.path.append('..')
# from util.Logger import Logger
# logger = Logger('log_loss')

def loss_func(out, gts, gts_mask, gts_covered, gtl, gtl_mask, gtl_covered):  #, covered):
    n_batch = gtl.shape[0]
    # width = gtl.shape[2]
    # height = gtl.shape[3]
    # shape = gtl.shape     # batch, depth, width, height
    loss = 0
    loss_gtl = 0
    loss_gtl = 0

    pred_s = torch.stack(out[0]).transpose(0,1).cpu()
    pred_l = torch.stack(out[1]).transpose(0,1).cpu()

    # 15, 3, 14, 45, 45
    ##################### loss with F score idea ##############

    # point
    thres = 0.01

    pred_s_uncovered = pred_s[gts_covered == False]
    gts_uncovered = gts[gts_covered == False]
    y_diff_s = torch.abs(pred_s_uncovered - gts_uncovered)
    y_diff_mean_s = torch.mean(y_diff_s)

    n_pred_s = torch.mean((pred_s_uncovered > thres).float())
    n_gt_s = torch.mean((gts_uncovered > thres).float())
    n_diff_mean_s = torch.abs(n_pred_s-n_gt_s)

    loss_gts = (1-y_diff_mean_s) * (1-n_diff_mean_s)
    loss_gts = - torch.log(loss_gts)
    loss += loss_gts

    ################################### link
    thres = 0.01

    # find active pixels of pred_l both covered and uncovered
    pred_l_abs = torch.abs(pred_l)
    x_index = [i*2 for i in range(gtl.shape[2])]
    y_index = [i*2+1 for i in range(gtl.shape[2])]
    
    x = pred_l_abs[:, :, x_index]
    y = pred_l_abs[:, :, y_index]
    n_i_active_x = x >= thres
    n_i_active_y = y >= thres
    n_i_active = n_i_active_x | n_i_active_y

    # get uncovered
    n_i_active_uncovered = n_i_active[gtl_covered==False]
    n_mean_pred_l = torch.mean(n_i_active_uncovered.float())


    # find active and uncovered
    gt_active_l = gtl_mask
    n_gt_active_uncovered = gt_active_l[gtl_covered==False]
    n_mean_gt_l = torch.mean(n_gt_active_uncovered.float())

    # find n_diff
    n_diff_mean_l = torch.abs(n_mean_pred_l - n_mean_gt_l)

    # find y_diff
    pred_l_uncovered = pred_l[gtl_covered==False]
    gt_l_uncovered = gtl[gtl_covered==False]
    y_diff = torch.abs(pred_l_uncovered - gt_l_uncovered)/2
    y_diff_mean = torch.mean(y_diff)
    loss_gtl = (1-y_diff_mean) * (1-n_diff_mean_l)
    loss_gtl = - torch.log(loss_gtl)
    loss += loss_gtl
    
    return loss, loss_gtl, loss_gtl

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
