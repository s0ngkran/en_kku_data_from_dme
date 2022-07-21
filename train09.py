import torch
import os
import numpy as np
import cv2
import json
from model04 import HandModel
from torch.utils.data import DataLoader
from dme_data_curriculum import DMEDataset
from lossfunc_to_control_covered_F_score_idea import loss_func

if __name__ == '__main__':
    ############################ config ###################
    TRAINING_JSON = 'training.curriculum.json'
    VALIDATION_JSON = 'validation.curriculum.json'
    BATCH_SIZE = 14
    SAVE_EVERY = 2
    LEARNING_RATE = 1e-4
    TRAINING_NAME = os.path.basename(__file__)
    NUM_WORKERS = 10
    LOG_FOLDER = 'log/'
    SAVE_FOLDER = 'save/'
    OPT_LEVEL = 'O2'
    CHECK_RUN = True

    # continue training
    IS_CONTINUE = False
    IS_CHANGE_LEARNING_RATE = False
    CONTINUE_PATH = './save/train09.pyepoch0000003702.model'
    NEW_LEARNING_RATE = 1e-4

    # check result
    IS_CHECK_RESULT = True
    TESTING_JSON = 'training.curriculum.json'
    DEVICE = 'cpu'
    TESTING_FOLDER = 'TESTING_FOLDER'
    WEIGHT_PATH = './save/train09.pyepoch0000006362.model'
    ############################################################
    print('starting...')



    if not IS_CHECK_RESULT:
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            # from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to run this example.")

        # load data json
        with open(TRAINING_JSON, 'r') as f:
            training_json = json.load(f)
        with open(VALIDATION_JSON, 'r') as f:
            validation_json = json.load(f)
    else:
        with open(TESTING_JSON, 'r') as f:
            testing_json = json.load(f)

    # manage batch
    def my_collate(batch):
        image_path, image, gts, gtl, gts_mask, gtl_mask, covered_point, covered_link = [],[],[],[],[],[],[],[]
        for i in batch:
            image_path.append(i['image_path'])
            image.append(torch.HalfTensor(i['image']))
            gts.append(i['gts'])
            gtl.append(i['gtl'])
            gts_mask.append(i['gts_mask'])
            gtl_mask.append(i['gtl_mask'])
            covered_point.append(i['covered_point'])
            covered_link.append(i['covered_link'])
        ans = {
            'image_path': image_path,
            'image': torch.cat(image),
            'gts': torch.cat(gts),
            'gtl': torch.cat(gtl),
            'gts_mask': torch.cat(gts_mask),
            'gtl_mask': torch.cat(gtl_mask),
            'covered_point': torch.cat(covered_point),
            'covered_link': torch.cat(covered_link),
        }
        return ans
    
    # load data
    if not IS_CHECK_RESULT:
        training_set = DMEDataset(training_json, test_mode=CHECK_RUN)
        validation_set = DMEDataset(validation_json, test_mode=CHECK_RUN)
        training_set_loader = DataLoader(training_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True) #, collate_fn=my_collate)
        validation_set_loader = DataLoader(validation_set,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True) #, collate_fn=my_collate)
    else:
        testing_set = DMEDataset(testing_json, test_mode=CHECK_RUN)
        testing_set_loader = DataLoader(testing_set,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True) #, collate_fn=my_collate)

    # init model
    channel = 3
    if not IS_CHECK_RESULT:
        model = HandModel(channel).to('cuda')
        optimizer = torch.optim.Adam(model.parameters())
        epoch = 0
    else:
        model = HandModel(channel)
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


    # train
    def train():
        global model, optimizer, epoch
        model.train()
        epoch += 1
        for iteration, dat in enumerate(training_set_loader):
            iteration += 1
            image = dat['image'].half().cuda()/255
            
            output = model(image)

            # print(output[0].shape, dat['gts'].shape, dat['gts_mask'].shape,
            #       dat['gtl'].shape, dat['gtl_mask'].shape)

            # print(dat['covered_point'].shape)
            # print(dat['covered_link'].shape)
            loss, loss_gts, loss_gtl = loss_func(
                output, dat['gts'], dat['gts_mask'], dat['covered_point'], dat['gtl'], dat['gtl_mask'], dat['covered_link'])
            if CHECK_RUN:
                print('loss_', loss.item())

            # if iteration%100 == 0:
            #     print(epoch, iteration, loss.item())

            write_loss(epoch, iteration, loss.item())
            write_loss_gts(epoch, iteration, loss_gts.item())
            write_loss_gtl(epoch, iteration, loss_gtl.item())

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

        if CHECK_RUN:
            print('loss',loss.item())

    def validation():
        global model
        model.eval()
        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            for iteration, dat in enumerate(validation_set_loader):
                iteration += 1
                image = dat['image'].half().cuda()/255

                output = model(image)

                loss_, loss_gts_, loss_gtl_ = loss_func(
                    output, dat['gts'], dat['gts_mask'], dat['covered_point'], dat['gtl'], dat['gtl_mask'], dat['covered_link'])
                if CHECK_RUN:
                    print('loss_', loss_)
                loss.append(loss_)
                loss_gts.append(loss_gts_)
                loss_gtl.append(loss_gtl_)

            loss = sum(loss)/len(loss)
            loss_gts = sum(loss_gts)/len(loss_gts)
            loss_gtl = sum(loss_gtl)/len(loss_gtl)
            write_loss_va(epoch, iteration, loss)
            write_loss_gts_va(epoch, iteration, loss_gts)
            write_loss_gtl_va(epoch, iteration, loss_gtl)
            if CHECK_RUN:
                print('va loss', loss)

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
            num_image = 0
            for iteration, dat in enumerate(testing_set_loader):
                iteration += 1
                print('iteration', iteration, len(testing_set_loader))
                # write original image
                _image = dat['image'] # img.shape == 14, ch, 360, 360
                if _image.shape[1] == 1:
                    for i, img in enumerate(_image):
                        img = np.array(img)
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_original.jpg'%i), img)

                else: # using io and transform
                    img_paths = dat['image_path']
                    for i, path in enumerate(img_paths):
                        img = cv2.imread(path)
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_original.jpg'%i), img)



                # write gtl image
                for i, gtl in enumerate(dat['gtl']):
                    for ii, img in enumerate(gtl):
                        img = img.mean(0).T
                        img = np.array(img) 
                        img = img*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_gtl.jpg'%(i, ii)), img)
                

                # write gts image
                for i, gts in enumerate(dat['gts']):
                    for ii, img in enumerate(gts):
                        img = img.max(0)[0].T
                        img = np.array(img)*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_gts.jpg'%(i, ii)), img)


                # manage before feed to model
                if DEVICE != 'cuda':
                    image = dat['image']/255
                else:
                    image = dat['image'].half().cuda()/255

                if image.shape[2] == 1:
                    image = image.unsqueeze(1) 
                # image size => batch, 3, 1, x, y

                output = model(image) # (s1,2,3), (l1,2,3)

                # write output
                s_group = output[0]
                l_group = output[1]

                for i, l in enumerate(l_group):
                    for ii, batch in enumerate(l):
                        img = batch.mean(0).T
                        img = np.array(img)*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_pred_gtl.jpg'%(i, ii)), img)
                for i, s in enumerate(s_group):
                    for ii, batch in enumerate(s):
                        img = batch.max(0)[0].T
                        img = np.array(img)*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_pred_gts.jpg'%(i, ii)), img)

            #     loss_, loss_gts_, loss_gtl_ = loss_func(
            #         output, dat['gts'], dat['gts_mask'], dat['covered_point'], dat['gtl'], dat['gtl_mask'], dat['covered_link'])
            #     loss.append(loss_)
            #     loss_gts.append(loss_gts_)
            #     loss_gtl.append(loss_gtl_)

            # loss = sum(loss)/len(loss)
            # loss_gts = sum(loss_gts)/len(loss_gts)
            # loss_gtl = sum(loss_gtl)/len(loss_gtl)

    # train
    while True:
        print('epoch', epoch)
        if not IS_CHECK_RESULT:
            train()
            validation()
        else:
            test()
            break

        if epoch == 1 or epoch % SAVE_EVERY == 0 and not IS_CHECK_RESULT:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'amp_state_dict': amp.state_dict(),
            }, SAVE_FOLDER + TRAINING_NAME + 'epoch%s.model' % (str(epoch).zfill(10)))
