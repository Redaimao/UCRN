import sys
from src import models

import pickle

from src.eval_metrics import *
from src.loss_functions import *
from src.utils import *
from torch.utils.tensorboard import SummaryWriter

import warnings

warnings.filterwarnings("ignore")


def test_initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    criterion = getattr(nn, hyp_params.criterion)()

    settings = {'model': model,
                'criterion': criterion,
                }
    return test_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def test_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    # # get model type
    # model = settings['model']
    # basic criterion:
    criterion = settings['criterion']

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = test_result(model, criterion, test_loader, hyp_params, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    sys.stdout.flush()
    input('New run input any key')


####################################################################
#
# testing manuscript
#
####################################################################

def test_result(model, criterion, loader, hyp_params, test=True):
    model.eval()
    total_loss = 0.0

    results = []
    truths = []
    last_h_ls = []
    uni_h_ls = []
    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

            batch_size = text.size(0)

            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, last_h_proj, h_list = net(text, audio, vision)
            total_loss += criterion(preds, eval_attr).item() * batch_size
            # Collect the results into dictionary
            results.append(preds)
            truths.append(eval_attr)
            last_h_ls.append(last_h_proj)
            uni_h_ls.append(h_list[:-1])

    avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

    results = torch.cat(results)
    truths = torch.cat(truths)

    save_vis_dir = os.path.join(hyp_params.vis_dir, hyp_params.name)
    if not os.path.exists(save_vis_dir):
        os.makedirs(save_vis_dir)
    lash_proj_arr = np.concatenate([np.array(batch_arr.cpu()) for batch_arr in last_h_ls], axis=0) #54 24 904*24
    print(len(uni_h_ls))
    l_feat_arr = np.concatenate([np.array(l_feat[0].cpu().reshape((-1, 30))) for l_feat in uni_h_ls], axis=0)
    a_feat_arr = np.concatenate([np.array(l_feat[1].cpu().reshape((-1, 30))) for l_feat in uni_h_ls], axis=0)
    v_feat_arr = np.concatenate([np.array(l_feat[2].cpu().reshape((-1, 30))) for l_feat in uni_h_ls], axis=0)
    print(truths.size())
    print(l_feat_arr.shape) #torch.Size([14, 24, 30])
    print(a_feat_arr.shape)
    print(v_feat_arr.shape)
    with open(save_vis_dir + '/last_proj_arr.pkl', 'wb') as file:
        pickle.dump([lash_proj_arr, np.array(results.cpu())], file)
    with open(save_vis_dir + '/uni_feat_arr.pkl', 'wb') as file:
        pickle.dump([l_feat_arr, a_feat_arr, v_feat_arr], file)

    return avg_loss, results, truths
