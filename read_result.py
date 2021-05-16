import json
import os
import numpy as np
import matplotlib.pyplot as plt


def read_loss(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
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
    epoch = np.array(epoch)
    loss = np.array(loss)
    return epoch, loss


def read_and_plot(filename, color):
    epoch, loss = read_loss(filename)
    plt.plot(epoch, loss, color)

# if __name__ == "__main__":
#     tr200 = 'train02.py.loss'
#     va200 = 'train02.py.loss_va'
#     tr500 = 'train03.py.loss'
#     va500 = 'train03.py.loss_va'
    
#     def get_140plus(epoch, loss):
#         c1 = epoch >= 140
#         c2 = epoch <= 340
#         cc = c1 * c2
        
#         epoch = epoch[cc]
    
#         loss = loss[cc]
#         avg = sum(loss)/len(loss)
#         return avg


#     epoch, loss = read_loss(tr200)
#     avg_tr200 = get_140plus(epoch, loss)
#     epoch, loss = read_loss(va200)
#     avg_va200 = get_140plus(epoch, loss)
#     epoch, loss = read_loss(tr500)
#     avg_tr500 = get_140plus(epoch, loss)
#     epoch, loss = read_loss(va500)
#     avg_va500 = get_140plus(epoch, loss)
#     print('tr200', avg_tr200)
#     print('va200', avg_va200)
#     print('tr500', avg_tr500)
#     print('va500', avg_va500)




if __name__ == '__main__':
    
    tr = input('training num (e.g. 1, 2, 3, ...) = ')
    read_and_plot('train%s.pyw.loss' % (str(tr).zfill(2)), 'r.')
    read_and_plot('train%s.pyw.loss_va' % (str(tr).zfill(2)), 'b-')
    read_and_plot('train%s.pyw.gts_loss' % (str(tr).zfill(2)), 'g.')
    read_and_plot('train%s.pyw.gtl_loss' % (str(tr).zfill(2)), 'yx')
    plt.show()
