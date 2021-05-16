import os
import cv2 
import torch
import numpy as np
import time
import torch.optim as optim
import time
from torchvision.utils import save_image
from datetime import datetime as dt
import copy

import sys
sys.path.append('..')
from util.torch2img import torch2img as t2img
from util.Logger import Logger

from util.data import data015_480_bigger as Data
from model03 import HandModel
from loss_function import loss_func
class Config:
    def __init__(self):
        
        
        self.thisname = os.path.basename(__file__)
        print('starting...', self.thisname)
        
        # commit data015_480
        self.data = Data
        self.model = HandModel
        self.lossfunction = loss_func
        self.savefolder = 'save/'
        
        self.channel = 1
        self.lr = 0.0001
        self.run_batch = 2
        self.to_epoch = 9000
        self.va_every = 1
        self.seed = 1
        
        self.continue_training = None
        # self.continue_training = 'save/train04_epoch0000009000_seed1.pth'
        
        self.fast_loading = False # use va set for training (fast loading)
        self.small_data = False    # use 100 image to training
    
class Trainer(Config):
    def __init__(self, ):
        super().__init__()
        assert self.savefolder[-1] == '/'
        assert self.channel in [1,3]
        
        if not os.path.exists('save'):
            os.mkdir('save')
        if not os.path.exists('saveimg'):
            os.mkdir('saveimg')
        if not os.path.exists('temp'):
            os.mkdir('temp')
    

        self.train_n = self.thisname.split('train')[1].split('.py')[0]
        
        self.model = self.model(self.channel).cuda()
        self.optimizer = optim.Adam(self.model.parameters())
        logfile = 'tr%s'%(self.train_n)
        self.logname = logfile
        self.logger = Logger(logfile)
        self.start_epoch = 0
        self.epoch = 0
        self.times = 0
        self.torch2img = t2img(self.thisname)
        self.unpack_func = self.to_grayscale if self.channel==1 else self.to_3channel
        self.data = self.data(self.logger, test=self.fast_loading, small_size=self.small_data)
        ############### cut out nan #######################
        img = []
        gts = []
        gtl = []
        cov_p = []
        cov_l = []
        label = []
        dat = self.data.tr
        for i in range(len(self.data.tr.img)):
            if torch.isinf(dat.img[i].type(torch.float32)).any() or torch.isnan(dat.img[i].type(torch.float32)).any() or torch.isinf(dat.gts[i].type(torch.float32)).any() or torch.isnan(dat.gts[i].type(torch.float32)).any() or torch.isinf(dat.gtl[i].type(torch.float32)).any() or torch.isnan(dat.gtl[i].type(torch.float32)).any():
                pass
            else:
                img.append(dat.img[i])
                gts.append(dat.gts[i])
                gtl.append(dat.gtl[i])
                cov_p.append(dat.covered_point[i])
                cov_l.append(dat.covered_link[i])
                label.append(dat.label[i])
        self.data.tr.img = torch.stack(img)
        self.data.tr.gts = torch.stack(gts)
        self.data.tr.gtl = torch.stack(gtl)
        self.data.tr.covered_point = torch.stack(cov_p)
        self.data.tr.covered_link = torch.stack(cov_l)
        self.data.tr.label = label
        self.logger.write('cut out nan in data tr')
        ######################################
        
        self.n_tr = len(self.data.tr.img)
        self.lr_ori = copy.deepcopy(self.lr) # for logging
    def print_config(self):
        self.logger.write('###### config #####')
        self.logger.write('thisname', self.thisname)
        self.logger.write('savefolder', self.savefolder)
        self.logger.write('channel', self.channel)
        self.logger.write('learning rate setting', self.lr_ori)
        self.logger.write('learning rate', self.lr)
        self.logger.write('batch', self.run_batch)
        self.logger.write('to_epoch', self.to_epoch)
        self.logger.write('va_every', self.va_every)
        self.logger.write('seed', self.seed)
        self.logger.write('continue_training', self.continue_training)
        self.logger.write('training')
        self.logger.write('fast_loading', self.fast_loading)
        self.logger.write('small_data', self.small_data)
    
    def addtime(self):
        self.times += time.time()-self.t0
        self.t0 = time.time()
    def loadstage(self, stagefile):
        assert stagefile[-4:] =='.pth'
        checkpoint = torch.load(stagefile)
        self.model.load_state_dict(checkpoint['model_state_dict'])           #
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   #
        self.start_epoch = checkpoint['epoch']                              #
        self.times = checkpoint['time']                                   #                                     #
        self.logger.write('loaded model_stage')
    def savemodel(self, ):
        self.addtime()
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'time': self.times,
            }, self.savefolder+'train%s_epoch%s_seed%s.pth'%(str(self.train_n).zfill(2), str(self.epoch).zfill(10), self.seed))

        msg = 'epoch=%d saved, used_time=%dmins'%(self.epoch, self.times/60)
        self.logger.write(msg)
        self.t0 = time.time()
    def scale_lr(self):
        all_iter = self.n_tr//self.batch
        lr = self.lr_ori / all_iter
        self.change_lr(lr)
    def change_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.logger.write('learning rate to' + str(lr))
        self.lr = lr

    def init_training(self):
        if self.continue_training is not None:
            self.loadstage(stagefile=self.continue_training)
        else:
            self.scale_lr()
        
        self.epoch = self.start_epoch

        # set example image
        indices = torch.randperm(len(self.data.tr.img)) 
        self.ex_training_ind = indices[:self.batch]
        indices = torch.randperm(len(self.data.va.img))
        self.ex_validation_ind = indices[:self.batch]
        indices = torch.randperm(len(self.data.te.img)) 
        self.ex_testing_ind = indices[:self.batch]
        self.logger.write('tr_size'+str(len(self.data.tr.img)))
        self.logger.write('va_size'+str(len(self.data.va.img)))
        self.logger.write('te_size'+str(len(self.data.te.img)))

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        time0 = time.time()
        toggle = True
        n_data = len(self.data.tr.img)
        permutation = torch.randperm(n_data)
        last_ind = n_data # when perm[i:ii] the last index ii is length of data| no need to -1
        batch = self.batch
        all_iter = n_data//batch
        iteration = 0
        for i in range(0, n_data, batch):
            iteration += 1
            self.optimizer.zero_grad()
            ii = i+batch
            if ii > last_ind: # use full batch data => data(15), batch(4) => use12_(4,4,4) do not use_3
                break     
            indices = permutation[i:ii]
            
            ####################### skip nan ind #################
            # nan_inds = [4526, 6986]
            # for ind in indices:
            #     if ind in nan_inds:
            #         continue
            #############################################################

            self.img, self.gts, self.gtl, self.covered_point, self.covered_link = self.unpack(self.data.tr, indices)
            
            ################# check nan and inf ##########################
            if torch.isnan(self.img).any() or torch.isinf(self.img).any() or torch.isnan(self.gts).any() or torch.isinf(self.gts).any() or torch.isnan(self.gtl).any() or torch.isinf(self.gtl).any():
                self.logger.write('found nan or inf value')
                self.logger.write('img',self.img)
                self.logger.write('gts',self.gts)
                self.logger.write('gtl',self.gtl)
                continue
            ##############################################

            # import cv2
            # import numpy as np
            # print('----',gts.shape)
            # cv2.imwrite('temp_img.bmp',np.array(img[0][0].cpu())*255)
            # cv2.imwrite('temp_gts.bmp',np.array(gts[0].cpu().max(0)[0])*255)
            # 1/0
            # print('img=',self.img.shape)
            # input('inp')
            output = self.model(self.img)
            
            loss = self.lossfunction(output, self.gtl, self.gts, self.covered_point, self.covered_link)
            # del output
            # torch.cuda.empty_cache()
        
            
            loss.backward(retain_graph=True)
            self.loss = loss
            self.optimizer.step()

            if toggle: # show first loss
                toggle = False
                self.logger.write('first loss %.5f'%self.loss)
            

            if time.time() - time0 > 60: # log every minute
                time0 = time.time()
                msg = 'epoch_=%d iter_=%d/%d loss_=%.5f'%(self.epoch, iteration, all_iter, self.loss)
                self.logger.write(msg)
                
            if self.MODE == 'test_mode': 
                if iteration >= 3:
                    break

        self.logger.write('epoch=%d loss=%.5f'%(self.epoch, self.loss))

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            va_loss = []
            n_data = len(self.data.va.img)
            ind = [i for i in range(n_data)]
            iteration = 0
            last_ind = n_data # when perm[i:ii] the last index ii is length of data| no need to -1
            for i in range(0, n_data, self.batch):
                iteration += 1
                ii = i+self.batch
                if ii > last_ind: 
                    ii = last_ind # use all data eg. (5, 5, 5, 4)
                indices = ind[i:ii]

                self.img, self.gts, self.gtl, self.covered_point, self.covered_link = self.unpack(self.data.va, indices)
                
                output = self.model(self.img)
                va_loss_ = self.lossfunction(output, self.gtl, self.gts, self.covered_point, self.covered_link)
            
                va_loss.append(va_loss_)
                
                if self.MODE == 'test_mode': 
                    if iteration >= 3:
                        break
            va_loss = sum(va_loss)/len(va_loss)
            self.logger.write('epoch=%d loss_va=%.5f'%(self.epoch, va_loss) )
    def ex_img(self, data_set, ind, modename):
        ############# config ######################
        index_predL, index_predS = 1, 3
        #######################################
        self.model.eval()
        with torch.no_grad():
            n_data = self.batch

            self.img, self.gts, self.gtl, self.covered_point, self.covered_link = self.unpack(data_set, ind)

            self.out = self.model(self.img)
            self.out = (self.out[index_predL], self.out[index_predS])
        savename = self.thisname+'_'+modename+'_ep' + str(self.epoch).zfill(5)+'_seed'+str(self.seed).zfill(2)
        header_msg = modename + '_epoch %d'%self.epoch
        self.torch2img.genimg_(img=self.img, out=self.out, gts=self.gts, gtl=self.gtl, filename=savename, msg=header_msg, savefolder=None)

    def to_grayscale(self, img):
        # rgb_weight = [0.2989, 0.5870, 0.1140]
        # img shape =  y, x, ch # numpy array
        img = img.type(torch.float32).cuda()/255 # torch y,x,ch
        bgr_weight = torch.FloatTensor([0.1140, 0.5870, 0.2989]).type(torch.float32).cuda()
        img = torch.matmul(img, bgr_weight).unsqueeze(0).transpose(1,2) #y,x,ch #y,x #x,y
        return img
    
    def to_3channel(self, img):
        # img = torch.Size([480, 480, 3]) # y,x,ch
        img = img.type(torch.float32).cuda()/255
        img = img.transpose(1,2)
        return img

    def unpack(self, data_set, indices):
        img = torch.stack([self.unpack_func(data_set.img[i]) for i in indices])
        gts = torch.stack([data_set.gts[i] for i in indices]).type(torch.float32)
        gtl = torch.stack([data_set.gtl[i] for i in indices]).type(torch.float32)
        covered_point = torch.stack([data_set.covered_point[i] for i in indices])
        covered_link = torch.stack([data_set.covered_link[i] for i in indices])
        return img, gts, gtl, covered_point, covered_link
    
    def check_size(self):
        def check(data, name):
            self.logger.write('dataset=%s, img=%d, gts=%d, gtl=%d, cov_p=%d, cov_l=%d'%(name, len(data.img),len(data.gts),len(data.gtl),len(data.covered_point),len(data.covered_link)))
            assert len(data.img) == len(data.gts) == len(data.gtl) == len(data.covered_point) == len(data.covered_link)
        
        check(self.data.tr, 'tr')
        check(self.data.va, 'va')
        check(self.data.te, 'te')

    def run(self, batch=None, to_epoch=None, va_every=None, seed=None):
        if batch is None:
            batch = self.run_batch
        if to_epoch is None:
            to_epoch = self.to_epoch
        if va_every is None:
            va_every = self.va_every
        if seed is None:
            seed = self.seed
        
        self.logger.write('seed='+str(seed))
        self.MODE = 'train_mode'
        self.batch = batch
        self.init_training()
        self.seed = seed
        self.t0 = time.time()
        self.logger.write('''
        ##############
        start training
        ##############''')
        self.print_config()
        self.logger.write(dt.now())
        while self.epoch < to_epoch: # start epoch = 0
            time0 = time.time()
            self.epoch += 1
            self.train()
            time_ = time.time() - time0
            self.logger.write('used_time this tr '+str(time_/60)+'mins')
            
            time0 = time.time()
            if self.epoch % va_every == 0:
                self.validation()
                self.ex_img(self.data.tr, self.ex_training_ind, 'TR')
                self.ex_img(self.data.va, self.ex_validation_ind, 'VA')
                # self.ex_img(self.data.te, self.ex_testing_ind, 'TE')
                self.savemodel()
            time_ = time.time() - time0
            self.logger.write('used_time this va and ex_img '+str(time_/60)+'mins\n')
            
    def test_run(self):
        self.logger.write('''
        ################
        start test_run
        ################''')
        self.check_size()
        self.MODE = 'test_mode'
        to_epoch = 3
        self.batch = self.run_batch
        self.init_training()
        self.seed = 1
        self.t0 = time.time()
        self.logger.write('start training, test_mode')
        while self.epoch <= to_epoch: # start epoch = 0
            self.epoch += 1
            self.logger.write('start train')
            self.train() # full train small set(50) 3 time

            self.logger.write('start va')
            self.validation()

            self.logger.write('start ex_img')
            self.ex_img(self.data.tr, self.ex_training_ind, 'TR')
            self.ex_img(self.data.va, self.ex_validation_ind, 'VA')
            self.ex_img(self.data.te, self.ex_testing_ind, 'TE')
            self.logger.write('fin epoch'+str(self.epoch))
            self.savemodel()
            
    def test_out(self, weight_file):
        assert weight_file[-4:] == '.pth'
        ##################################
        savefolder = 'temp/'
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)
        ################################
        self.loadstage(weight_file)
        self.epoch = self.start_epoch-1 

        indices = torch.randperm(self.data.tr.size) + 1
        self.ex_training_ind = indices[:2] # for dummy stack
        self.model.eval()
        with torch.no_grad():
            img, gts, gtl = self.unpack(self.data.tr, self.ex_training_ind)
            out1, out2, out3, out4 = self.model(img) 
            
            # batch, part, x, y
            out1 = out1[0].mean(0)
            out2 = out2[0].mean(0)
            out3 = out3[0].mean(0)
            out4 = out4[0].max(0)[0]

            self.save_img(out1, savefolder+'out1')
            self.save_img(out2, savefolder+'out2')
            self.save_img(out3, savefolder+'out3')
            self.save_img(out4, savefolder+'out4')

    def save_img(self, tensor, savefile):
        assert len(tensor.shape) == 2
        save_image(tensor.transpose(0,1), savefile+'.bmp')
        print('saved', savefile+'.bmp')


if __name__ == "__main__":
    trainer = Trainer()
    trainer.test_run()
    trainer.run()
 