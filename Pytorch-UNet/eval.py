import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from myloss import dice_coeff
from utils import dense_crf


def eval_net(net, loader, device, gpu=False):
    if loader.dataset.train:
      print('Checking accuracy on training set')
    else:
        print('Checking accuracy on test set')   

    tot = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float32)

            if gpu:
                X = Variable(X, requires_grad=True).cuda()
                y = Variable(y, requires_grad=True).cuda()
            else:
                X = Variable(X, requires_grad=True)
                y = Variable(y, requires_grad=True)

            X.unsqueeze_(1)
            y.unsqueeze_(1)
            y_pred = net(X)

            y_pred = (F.sigmoid(y_pred) > 0.6).float()
            # print("y_pred.shape")
            # print(y_pred.shape)
            # print("y.shape")
            # print(y.shape)
            # y_pred = F.sigmoid(y_pred).float()

            dice = dice_coeff(y_pred, y.float()).data[0]
            tot += dice

            if 0:
                X = X.data.squeeze(0).cpu().numpy()
                X = np.transpose(X, axes=[1, 2, 0])
                y = y.data.squeeze(0).cpu().numpy()
                y_pred = y_pred.data.squeeze(0).squeeze(0).cpu().numpy()
                print(y_pred.shape)

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 4, 1)
                ax1.imshow(X)
                ax2 = fig.add_subplot(1, 4, 2)
                ax2.imshow(y)
                ax3 = fig.add_subplot(1, 4, 3)
                ax3.imshow((y_pred > 0.5))

                Q = dense_crf(((X * 255).round()).astype(np.uint8), y_pred)
                ax4 = fig.add_subplot(1, 4, 4)
                print(Q)
                ax4.imshow(Q > 0.5)
                plt.show()
    return tot / i
