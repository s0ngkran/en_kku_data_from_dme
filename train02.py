import torch
import os 
import cv2
import json
from model02 import HandModel
from torch.utils.data import DataLoader
from dme_data import DMEDataset
from lossfunc import loss_func

BATCH_SIZE = 20
SAVE_EVERY = 1
LEARING_RATE = 0.00001
TRAINING_NAME = os.path.basename(__file__)
LOG_FOLDER = 'log/'
SAVE_FOLDER = 'save/'

try:
    # from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    # from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# load data json
with open('training_json', 'r') as f:
    training_json = json.load(f)
with open('validation_json', 'r') as f:
    validation_json = json.load(f)

# load data
training_set = DMEDataset(training_json[:2400], test_mode=False)
validation_set = DMEDataset(validation_json, test_mode=False)
training_set_loader = DataLoader(
    training_set, batch_size=BATCH_SIZE, num_workers=5, shuffle=True)
validation_set_loader = DataLoader(
    validation_set,  batch_size=BATCH_SIZE, num_workers=5, shuffle=True)

# load model
channel=1
model = HandModel(channel).to('cuda')
optimizer = torch.optim.Adam(model.parameters())

update_per_epoch = len(training_set_loader)/BATCH_SIZE
learning_rate = LEARING_RATE/update_per_epoch
for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
epoch = 0

# write loss value
def write_loss(epoch, iteration, loss):
    with open(LOG_FOLDER + TRAINING_NAME + '.loss', 'a') as f:
        f.write('epoch=%d,iter=%d,loss=%f\n'%(epoch, iteration, loss))
def write_loss_gts(epoch, iteration, loss):
    with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss', 'a') as f:
        f.write('epoch=%d,iter=%d,loss=%f\n'%(epoch, iteration, loss))
def write_loss_gtl(epoch, iteration, loss):
    with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss', 'a') as f:
        f.write('epoch=%d,iter=%d,loss=%f\n'%(epoch, iteration, loss))

def write_loss_va(epoch, iteration, loss):
    with open(LOG_FOLDER + TRAINING_NAME + '.loss_va', 'a') as f:
        f.write('epoch=%d,iter=%d,loss=%f\n'%(epoch, iteration, loss))
def write_loss_gts_va(epoch, iteration, loss):
    with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss_va', 'a') as f:
        f.write('epoch=%d,iter=%d,loss=%f\n'%(epoch, iteration, loss))
def write_loss_gtl_va(epoch, iteration, loss):
    with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss_va', 'a') as f:
        f.write('epoch=%d,iter=%d,loss=%f\n'%(epoch, iteration, loss))


# train
def train():
    global model, optimizer, epoch
    model.train()
    epoch += 1
    for iteration, dat in enumerate(training_set_loader):
        iteration += 1
        image = dat['image'].half().cuda()/255
        image = image.unsqueeze(1)
        
        output = model(image)

        # print('---', 'gtl', dat['gtl'].shape, 'gts', dat['gts'].shape,
        #       'co point', dat['covered_point'].shape, 'co link', dat['covered_link'].shape)
        loss, loss_gts, loss_gtl = loss_func(output, dat['gtl'], dat['gts'], dat['covered_point'], dat['covered_link'])
        write_loss(epoch, iteration, loss.item())
        write_loss_gts(epoch, iteration, loss_gts.item())
        write_loss_gtl(epoch, iteration, loss_gtl.item())

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

def validation():
    global model
    model.eval()
    with torch.no_grad():
        loss, loss_gts, loss_gtl = [],[],[]
        for iteration, dat in enumerate(validation_set_loader):
            iteration += 1
            image = dat['image'].half().cuda()/255
            image = image.unsqueeze(1)

            output = model(image)

            loss_, loss_gts_, loss_gtl_ = loss_func( output, dat['gtl'], dat['gts'], dat['covered_point'], dat['covered_link'])
            loss.append(loss_)
            loss_gts.append(loss_gts_)
            loss_gtl.append(loss_gtl_)
            
        loss = sum(loss)/len(loss)
        loss_gts = sum(loss_gts)/len(loss_gts)
        loss_gtl = sum(loss_gtl)/len(loss_gtl)
        write_loss_va(epoch, iteration, loss)
        write_loss_gts_va(epoch, iteration, loss_gts)
        write_loss_gtl_va(epoch, iteration, loss_gtl)

while True:
    print('epoch',epoch)
    train()
    validation()
    if epoch == 1 or epoch % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVE_FOLDER + TRAINING_NAME + 'epoch%s.model'%(str(epoch).zfill(10)))
        


