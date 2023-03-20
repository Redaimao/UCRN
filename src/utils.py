import torch
import os
from src.dataset import Multimodal_Datasets
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot



def get_data(args, dataset, split='train'):
    print('dataset name:', dataset)
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'

    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        if args.use_bert:
            print("bert dataset not founded!!!")
        else:
            data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
            torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model


"""
Visualization functions:
"""


def plot_loss(train_ls, val_ls, test_ls, save_dir, model_name):
    host = host_subplot(111)
    # plt.subplots_adjust(right=0.8)
    if train_ls is not None:
        # set labels for train loss
        host.set_xlabel('epochs')
        host.set_ylabel('loss')

        # plot curves
        plt.xticks(list(range(len(train_ls))))
        p1, = host.plot(list(range(len(train_ls))), train_ls, label="training loss")
        # p1, = host.plot([0, 1, 2, 3], train_ls, label="training loss")

        # set label color
        host.axis["left"].label.set_color(p1.get_color())
        if val_ls is not None:
            # par1 = host.twinx()
            p2, = host.plot(list(range(len(val_ls))), val_ls, label="validation loss")
            host.axis["left"].label.set_color(p2.get_color())

        # set location of the legend,
        # 1->rightup corner, 2->leftup corner, 3->leftdown corner
        # 4->rightdown corner, 5->rightmid ...
        host.legend(loc=1)
        plt.draw()
        plt.show()
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                plt.savefig(save_dir + '/' + model_name + '.png')
                print(f"Saved model at /{save_dir + '/' + model_name}.png!")


def plot_acc_loss(loss, acc):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()  # share x-axis

    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("test-loss")
    par1.set_ylabel("test-accuracy")

    # plot curves
    p1, = host.plot(range(len(loss)), loss, label="loss")
    p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()


def plotCurve(x_vals, y_vals,
              x_label, y_label,
              x2_vals=None, y2_vals=None,
              legend=None,
              figsize=(3.5, 2.5)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')

    if legend:
        plt.legend(legend)


"""
plotCurve(range(1, num_epochs + 1), train_ls,
      "epoch", "loss",
      range(1, num_epochs + 1), test_ls,
      ["train", "test"]
     )
"""

if __name__ == '__main__':
    # loss_ls = [0.02, 0.03, 0.04, 0.05]
    # acc_ls = [0.67, 0.78, 0.80, 0.85]
    # # plot_acc_loss(loss_ls, acc_ls)
    # plot_loss(loss_ls, acc_ls, None, None,None)

    # a = torch.randn(2, 3, 4)
    # print('a:', a)
    # b = torch.randn(2, 4, 3)
    # print('b:', b)
    #
    # c = torch.bmm(a, b)
    # print(c)
    # print(c.size())
    # d = torch.diagonal(c, dim1=-2, dim2=-1)
    # print(d)
    # print(d.size())
    #
    # bs, _, _ = c.size()
    # c = c.view(bs, -1)
    # c = torch.sum(c, 1)
    # print(c)
    # print(c.size())

    import numpy as np

    # test_truth = [-1, 0, 0, 2, 1]
    # non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not True)])
    # print(non_zeros)

    n = 5
    batch_size = 4
    arr = np.array([2, 2, -1, -2.5])
    brr = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    brr = brr[np.where(arr >= 2)]
    print(brr)

    arr1 = arr[np.where(arr >= 2)]

    print(arr1)
