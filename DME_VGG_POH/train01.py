import torch
import os
import numpy as np
import cv2
import json
import time
import sys
from model_vgg_poh import VGG_PoH as Model
from torch.utils.data import DataLoader
from dme_dataset import Dataset_DME_VGG_PoH as MyDataset
import torch.nn.functional as F

# from lossfunc_to_control_covered_F_score_idea import loss_func
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
from argparse import ArgumentParser

if __name__ == '__main__':
    # python train01.py -sh -ck
    parser = ArgumentParser()
    parser.add_argument('-sh', '--show', help='show loss value on training and validation', action='store_true')
    parser.add_argument('-ck', '--check_run', help='run only first 50 sample', action='store_true') 
    parser.add_argument('-co', '--continue_save', help='continue at specific epoch', type=int) 
    parser.add_argument('-b', '--batch_size', help='set batch size', type=int) 
    parser.add_argument('-te', '--test', nargs=2, help='[EPOCH_NUM, JSON_SET] check result at specific epoch; JSON_SET can be "tr","va","te" ') 

    parser.add_argument('-bt', '--bootstrap', help='train with bootstrap', action='store_true')
    parser.add_argument('-col', '--continue_last', help='continue at last epoch', action='store_true')
    parser.add_argument('-xx', '--xx', help='train with bootstrap', action='store_true')
    parser.add_argument('-nw', '--n_worker', help='n_worker', type=int)
    args = parser.parse_args()
    print(args)

    ############################ config ###################
    JSON_PATTERN = '../XXX_json'
    TRAINING_JSON = JSON_PATTERN.replace('XXX', 'training')
    VALIDATION_JSON = JSON_PATTERN.replace('XXX', 'validation')
    BATCH_SIZE = 20 if args.batch_size == None else args.batch_size
    SAVE_EVERY = 2
    LEARNING_RATE = 1e-3
    TRAINING_NAME = os.path.basename(__file__)
    N_WORKERS = args.n_worker if args.n_worker != None else 10
    LOG_FOLDER = 'log/'
    SAVE_FOLDER = 'save/'
    OPT_LEVEL = 'O2'
    CHECK_RUN = args.check_run
    IS_BOOTSTRAP = args.bootstrap
    AMP_ENABLED = False

    # continue training
    IS_CONTINUE = False if args.continue_save is None and args.continue_last == False else True
    # CONTINUE_PATH = './save/train09.pyepoch0000003702.model'
    if args.continue_last == False:
        continue_epoch = args.continue_save
        CONTINUE_PATH = './%s/%sepoch%s.model'%(SAVE_FOLDER, TRAINING_NAME,  str(continue_epoch).zfill(10))
    else:
        def find_last_epoch_path():
            for _,_,fname_list in os.walk(SAVE_FOLDER):
                print('walk')
            
            name_list = []
            for fname in fname_list:
                if fname.startswith(TRAINING_NAME):
                    name_list.append(fname)
            name_list.sort()
            path = os.path.join(SAVE_FOLDER, name_list[-1])
            return path
        CONTINUE_PATH = find_last_epoch_path()
        print('continue on last epoch')
        print(CONTINUE_PATH)
        print()

    IS_CHANGE_LEARNING_RATE = False
    NEW_LEARNING_RATE = 1e-4

    # check result
    if args.test is not None:
        # check type
        try:
            int(args.test[0])
            _ = {
                'tr': 'training_set',
                'va': 'validation_set',
                'te': 'testing_set',
                'ten': 'testing_set_new',
            }
            assert args.test[1] in _.keys()
            args.test[1] = _[args.test[1]]
        except:
            pass

    IS_CHECK_RESULT = False if args.test is None else True
    TESTING_JSON = JSON_PATTERN.replace('XXX', args.test[1]) if args.test is not None else 'nothing'
    print(TESTING_JSON,'-----as testing_set')
    DEVICE = 'cpu'
    TESTING_FOLDER = 'TESTING_FOLDER/'
    # WEIGHT_PATH = './save/train09.pyepoch0000003702.model'
    if args.test is not None:
        WEIGHT_PATH = './%s/%sepoch%s.model'%(SAVE_FOLDER, TRAINING_NAME, str(args.test[0]).zfill(10))
    ############################################################

    print('starting...')
    for folder_name in [LOG_FOLDER, SAVE_FOLDER, TESTING_FOLDER]:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    if not IS_CHECK_RESULT and AMP_ENABLED:
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            # from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to run this example.")

    # manage batch
    def my_collate(batch):
        image_path, ground_truth, hand_landmarks = [],[],[]
        for item in batch:
            image_path.append(item['img_path'])
            ground_truth.append(item['ground_truth'])
            # hand_landmarks.append(item['hand_landmarks'])

        ans = {
            'img_path':image_path, 
            'ground_truth':ground_truth, 
            # 'hand_landmarks': hand_landmarks,
        }
        return ans
    
    # load data
    if not IS_CHECK_RESULT:
        training_set = MyDataset(TRAINING_JSON, test_mode=CHECK_RUN)
        validation_set = MyDataset(VALIDATION_JSON, test_mode=CHECK_RUN)
        training_set_loader = DataLoader(training_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True, drop_last=True) #, collate_fn=my_collate)
        validation_set_loader = DataLoader(validation_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)
    else:
        testing_set = MyDataset(TESTING_JSON, test_mode=CHECK_RUN)
        testing_set_loader = DataLoader(testing_set,  batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)

    # init model
    channel = 3
    if not IS_CHECK_RESULT:
        # model = Model(channel).to('cuda')
        model = Model().to('cuda')
        optimizer = torch.optim.Adam(model.parameters())
        epoch = 0
    else:
        # model = Model(channel)
        model = Model()
        epoch = 0
    
    # load state
    if not IS_CHECK_RESULT:
        if IS_CONTINUE:
            checkpoint = torch.load(CONTINUE_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # amp.load_state_dict(checkpoint['amp_state_dict'])
            epoch = checkpoint['epoch']
            if IS_CHANGE_LEARNING_RATE:
                # scale learning rate
                update_per_epoch = len(training_set_loader)/BATCH_SIZE
                learning_rate = NEW_LEARNING_RATE/update_per_epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
        else:
            # scale learning rate
            update_per_epoch = len(training_set_loader)/BATCH_SIZE
            learning_rate = LEARNING_RATE/update_per_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        if AMP_ENABLED:
            # init amp
            print('initing... amp')
            model, optimizer = amp.initialize(model, optimizer, opt_level=OPT_LEVEL)
    else:
        checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])



    # write loss value
    def write_loss(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gts(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gtl(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gts_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gtl_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))

    def get_first_random_from_loader(loader):
        for dat in training_set_loader:
            break
        return dat

    # train
    def train():
        global model, optimizer, epoch
        model.train()
        epoch += 1
        for iteration, dat in enumerate(training_set_loader):
            if IS_BOOTSTRAP:
                dat = get_first_random_from_loader(training_set_loader)
            iteration += 1
            inp = dat['img'].cuda()
            
            optimizer.zero_grad()
            output = model(inp)

            gt = dat['ground_truth']
            gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()

            # print(output[0].shape, dat['gts'].shape, dat['gts_mask'].shape,
            #       dat['gtl'].shape, dat['gtl_mask'].shape)

            # print(dat['covered_point'].shape)
            # print(dat['covered_link'].shape)
            # print(output, gt)
            # def loss_func(a, b):
            #     return torch.tensor([1])
            loss = loss_func(output, gt)
            if CHECK_RUN:
                pass
                # print('iter', iteration,'loss_', loss.item())

            # if iteration%100 == 0:
            #     print(epoch, iteration, loss.item())

            # write_loss(epoch, iteration, loss.item())
            # write_loss_gts(epoch, iteration, loss_gts.item())
            # write_loss_gtl(epoch, iteration, loss_gtl.item())

            if AMP_ENABLED:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        if CHECK_RUN:
            print('ep', epoch, 'loss',loss.item())
            print('''
            ################################
            #                              #
            #                              #
            #    this is checking mode     #
            #                              #
            #                              #
            ################################
            ''')
        if args.show:
            print('ep', epoch, '---loss- %.6f'%loss.item())

    def validation():
        global model
        model.eval()
        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            for iteration, dat in enumerate(validation_set_loader):
                iteration += 1
                inp = dat['hand_landmarks'].cuda()
                
                output = model(inp)

                gt = dat['ground_truth']
                gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()

                loss_ = loss_func(output, gt)

                if CHECK_RUN:
                    print('loss_', loss_)
                loss.append(loss_)

            loss = sum(loss)/len(loss)
            write_loss_va(epoch, iteration, loss)
            if CHECK_RUN:
                print('va loss', loss)
                time.sleep(0.3)
            if args.show:
                print('ep', epoch, '-----------va- %.6f'%loss)

    def test():
        global model
        model.eval()

        # mk folder
        if not os.path.exists(TESTING_FOLDER):
            os.mkdir(TESTING_FOLDER)
        else:
            os.system('rm -r %s'%TESTING_FOLDER)
            os.mkdir(TESTING_FOLDER)

        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            n_correct = 0
            n_fail = 0
            for iteration, dat in enumerate(testing_set_loader):
                iteration += 1
                inp = dat['hand_landmarks'].cpu()
                
                output = model(inp)

                gt = dat['ground_truth']
                gt = torch.tensor([int(i) for i in gt], dtype=torch.long)
                if args.show:
                    for b, _gt in zip(output, gt):

                        res = 'correct' if torch.argmax(b) == _gt else '************* fail'
                        if res == 'correct':
                            n_correct += 1
                        else:
                            n_fail += 1
                        print('output', b,'\n\n-> pred', torch.argmax(b).item(), ' gt', _gt.item(), res)
                        print()


                loss_ = loss_func(output, gt)

                if CHECK_RUN:
                    print('loss_', loss_)
                loss.append(loss_)
            total = len(testing_set_loader)
            if n_correct + n_fail != total:
                print('''
###################################################
         warning n_correct + n_fail != total
##################################################
                        ''')
                print('n_correct+n_fail=', n_correct +n_fail)
                print('len',total)
                total = n_correct+n_fail

            
            print('\n\ntotal', total)
            print('n_correct', n_correct)
            print('n_fail', n_fail)
            acc = n_correct/total*100
            print('accuracy %.2f'% (acc))


            loss = sum(loss)/len(loss)
            write_loss_va(epoch, iteration, loss)
            if CHECK_RUN:
                print('te loss', loss)
                time.sleep(0.3)
            if args.show:
                print('ep', epoch, '-----------te- %.6f'%loss)  

                

    # train
    while True:
        if not IS_CHECK_RESULT:
            train()
            
            if epoch == 1 or epoch % SAVE_EVERY == 0 and not IS_CHECK_RESULT:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'amp_state_dict': amp.state_dict(),
                }, SAVE_FOLDER + TRAINING_NAME + 'epoch%s.model' % (str(epoch).zfill(10)))

                # validate if model is saved
                validation()
                if CHECK_RUN or args.show:
                    print('* saved ep', epoch)
        else:
            test()
            break

