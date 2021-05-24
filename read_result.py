import json
import os
import numpy as np
import matplotlib.pyplot as plt


def get_lastest_index(epoch):
    epoch_ = epoch[:-1]
    shift = epoch[1:]

    diff = epoch_ - shift
    try:
        index = np.where(diff>0)[0][-1] + 1
    except:
        index = 0
    return index


def read_loss(filename, **kwargs):
    with open(filename, 'r') as f:
        data = f.readlines()
    # splite data
    epoch, iteration, loss = [], [], []
    for dat in data:
        epoch_, iteration_, loss_ = dat.split(',')
        epoch_ = epoch_.split('epoch=')[1]
        iteration_ = iteration_.split('iter=')[1]
        loss_ = loss_.split('loss=')[1]
        epoch_, iteration_, loss_ = int(epoch_), int(iteration_), float(loss_)
        epoch.append(epoch_)
        iteration.append(iteration_)
        loss.append(loss_)
    
    # convert to array
    epoch = np.array(epoch)
    loss = np.array(loss)

    # select lastest update
    if kwargs.get('last_update'):
        index = get_lastest_index(epoch)
        epoch = epoch[index:]
        loss = loss[index:]
    return epoch, loss


def read_and_plot(filename, color, **kwargs):
    epoch, loss = read_loss(filename, **kwargs)
    plt.plot(epoch, loss, color)

def custom_plot():
    tr200 = 'train02.py.loss'
    va200 = 'train02.py.loss_va'
    tr500 = 'train03.py.loss'
    va500 = 'train03.py.loss_va'
    tr900 = 'train04.py.loss'
    va900 = 'train04.py.loss_va'
    tr1300 = 'train05.py.loss'
    va1300 = 'train05.py.loss_va'
    
    def get_140plus(epoch, loss):
        c1 = epoch >= 140
        c2 = epoch <= 340
        cc = c1 * c2
        
        epoch = epoch[cc]
    
        loss = loss[cc]
        avg = sum(loss)/len(loss)
        return avg

    def get_avg(file):
        file = './log/' + file
        epoch, loss = read_loss(file)
        avg = get_140plus(epoch, loss)
        print(avg)
        return avg

    tr = [get_avg(i) for i in [tr200, tr500, tr900, tr1300]]
    va = [get_avg(i) for i in [va200, va500, va900, va1300]]
    tr = np.array(tr)
    va = np.array(va)

    for tr_, va_ in zip(tr, va):
        print(va_/ tr_)


    plt.plot(tr, '-r')
    plt.plot(tr, 'sr')
    plt.plot(va, '-b')
    plt.plot(va, 'sb')
    plt.show()


def main(**kwargs):
    tr = input('training num (e.g. 1, 2, 3, ...) = ')
    save_folder = 'log/'
    read_and_plot(save_folder + 'train%s.py.loss' % (str(tr).zfill(2)), 'r.', **kwargs)
    read_and_plot(save_folder + 'train%s.py.loss_va' % (str(tr).zfill(2)), 'b-', **kwargs)
    read_and_plot(save_folder + 'train%s.py.gts_loss' % (str(tr).zfill(2)), 'g.', **kwargs)
    read_and_plot(save_folder + 'train%s.py.gtl_loss' % (str(tr).zfill(2)), 'yx', **kwargs)
    plt.show()

if __name__ == '__main__':
    main(last_update=False)
    # custom_plot()
