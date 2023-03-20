# jsd loss function definition
import torch.nn.functional as F
import torch.nn as nn


class MultiJSD(nn.Module):
    def __init__(self):
        super(MultiJSD, self).__init__()

    def m_jsd(self, tgt_p, tgt_q, get_softmax=True):
        if get_softmax:
            tgt_p = F.softmax(tgt_p)
            tgt_q = F.softmax(tgt_q)

        tgt_mean_log = (tgt_p + tgt_q) / 2
        js_loss = (F.kl_div(tgt_mean_log, tgt_p, reduction='batchmean', log_target=True)
                   + F.kl_div(tgt_mean_log, tgt_q, reduction='batchmean', log_target=True)) / 2

        return js_loss

    def forward(self, h_list, get_softmax=False):
        _, batch_size, _ = h_list[0].size()
        if len(h_list) == 3:
            mjsd_loss = (self.m_jsd(h_list[0], h_list[1], get_softmax)
                         + self.m_jsd(h_list[1], h_list[2], get_softmax)
                         + self.m_jsd(h_list[2], h_list[0], get_softmax)) / 3
        elif len(h_list) == 2:
            mjsd_loss = self.m_jsd(h_list[0], h_list[1])
        else:
            mjsd_loss = 0.0
        return mjsd_loss / batch_size
