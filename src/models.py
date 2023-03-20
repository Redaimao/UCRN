"""
author: bao
date: 2020-11-04
functionality: multimodal sequence fusion models
Model names: UCRN
"""

import torch
from modules.transformer import TransformerEncoder
from modules.transformer import SQLayer
from src.loss_functions import *


class UCRNModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a UCRN model. Unimodal refinement module + self quality improvement layers +
        cross modal fusion module + cross modal JS divergence loss
        """
        super(UCRNModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.temp_proj_dim = hyp_params.temp_proj

        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        self.partial_mode = self.lonly + self.aonly + self.vonly
        combined_dim, combined_type = self.get_combined_dim_and_type()

        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 1*: temporal projection layer: project temporal dim to the same dim:
        self.temp_proj_l = nn.Conv1d(self.orig_l_len, self.temp_proj_dim, kernel_size=1, padding=0, bias=False)
        self.temp_proj_a = nn.Conv1d(self.orig_a_len, self.temp_proj_dim, kernel_size=1, padding=0, bias=False)
        self.temp_proj_v = nn.Conv1d(self.orig_v_len, self.temp_proj_dim, kernel_size=1, padding=0, bias=False)

        # 2. Uni-modal refinement module: self attention using transformer layer
        # 3. and self quality improvement layers
        if 'l' in list(combined_type):
            self.l_refine = self.get_network(self_type='l_refine')
            self.l_quality = self.get_quality_network(self_type='l_quality')
        if 'a' in list(combined_type):
            self.a_refine = self.get_network(self_type='a_refine')
            self.a_quality = self.get_quality_network(self_type='a_quality')
        if 'v' in list(combined_type):
            self.v_refine = self.get_network(self_type='v_refine')
            self.v_quality = self.get_quality_network(self_type='v_quality')

        # 4. Cross-modal Fusion
        # trimodal inputs
        self.cross_modal_fusion = self.get_network(self_type='cross_fusion')

        # 5. Projection layers: for classification or regression
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_combined_dim_and_type(self):
        """
        Check and get the combination dimensionality of inputs modalities; indicate the combination type
        :return:
        """
        if self.partial_mode == 3:
            combined_dim = self.d_l + self.d_a + self.d_v
            combined_type = 'lav'
        elif self.partial_mode == 2 and not self.lonly:
            combined_dim = self.d_v + self.d_a
            combined_type = 'av'
        elif self.partial_mode == 2 and not self.aonly:
            combined_dim = self.d_l + self.d_v
            combined_type = 'lv'
        elif self.partial_mode == 2 and not self.vonly:
            combined_dim = self.d_l + self.d_a
            combined_type = 'la'
        elif self.partial_mode == 1 and self.lonly:
            combined_dim = self.d_l
            combined_type = 'l'
        elif self.partial_mode == 1 and self.aonly:
            combined_dim = self.d_a
            combined_type = 'a'
        elif self.partial_mode == 1 and self.vonly:
            combined_dim = self.d_v
            combined_type = 'v'
        else:
            raise ValueError("unknown partial mode type")

        return combined_dim, combined_type

    def get_quality_network(self, self_type='l_quality'):
        """
        get self-quality improvement layers according to input modality
        :param self_type:
        :return:
        """
        if self_type in ['l_quality', ]:
            embed_dim = self.d_l
        elif self_type in ['a_quality']:
            embed_dim = self.d_a
        elif self_type in ['v_quality']:
            embed_dim = self.d_v
        else:
            raise ValueError("unknown quality network type")
        return SQLayer(embed_dim, reduction=16)

    def get_network(self, self_type='l', layers=-1):
        """
        get single modality refinement module: transformer encoder network
        :param self_type:
        :param layers:
        :return:
        """
        # unimdal refinement modules
        combined_dim, combined_type = self.get_combined_dim_and_type()

        if self_type in ['l_refine', ]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['v_refine', ]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['a_refine', ]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a

        elif self_type in ['cross_fusion']:
            embed_dim, attn_dropout = combined_dim, self.attn_dropout

        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def temp_projection(self, x_temp, x_type='l'):

        if self.temp_proj_dim == 0:
            if type(x_temp) == tuple:
                x_temp = x_temp[0]
            x_temp = x_temp[-1]
        else:
            if x_type == 'l':
                x_temp = (self.temp_proj_l(x_temp.permute(1, 0, 2))).permute(1, 0, 2)
            elif x_type == 'a':
                x_temp = (self.temp_proj_a(x_temp.permute(1, 0, 2))).permute(1, 0, 2)
            elif x_type == 'v':
                x_temp = (self.temp_proj_v(x_temp.permute(1, 0, 2))).permute(1, 0, 2)
            else:
                raise ValueError("unknown temp projection type!")
        return x_temp

    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        combined_dim, combined_type = self.get_combined_dim_and_type()
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)  # torch.Size([24, 300, 50])
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        h_list = []
        if 'l' in list(combined_type):
            h_l_refine = self.l_refine(proj_x_l)  # torch.Size([50, 32, 30]), timestep, batchsize, n_feature
            h_l_refine = self.temp_projection(h_l_refine, 'l')  # torch.Size([1, 32, 30])
            h_l_quality = self.l_quality(h_l_refine)  # torch.Size([1, 32, 30])
            h_list.append(h_l_refine)
        if 'a' in list(combined_type):
            h_a_refine = self.a_refine(proj_x_a)  # torch.Size([500, 32, 30])
            h_a_refine = self.temp_projection(h_a_refine, 'a')  # torch.Size([1, 32, 30])
            h_a_quality = self.a_quality(h_a_refine)  # torch.Size([1, 32, 30])
            h_list.append(h_a_refine)
        if 'v' in list(combined_type):
            h_v_refine = self.v_refine(proj_x_v)  # torch.Size([500, 32, 30])
            h_v_refine = self.temp_projection(h_v_refine, 'v')  # torch.Size([1, 32, 30])
            h_v_quality = self.v_quality(h_v_refine)  # torch.Size([1, 32, 30])
            h_list.append(h_v_refine)

        if self.partial_mode == 3 and combined_type == 'lav':
            h_fusion = self.cross_modal_fusion(torch.cat([h_l_quality, h_a_quality, h_v_quality], dim=2))

        elif self.partial_mode == 2:
            if combined_type == 'la':
                h_fusion = self.cross_modal_fusion(torch.cat([h_l_quality, h_a_quality], dim=2))
            if combined_type == 'lv':
                h_fusion = self.cross_modal_fusion(torch.cat([h_l_quality, h_v_quality], dim=2))
            if combined_type == 'av':
                h_fusion = self.cross_modal_fusion(torch.cat([h_a_quality, h_v_quality], dim=2))
        elif self.partial_mode == 1:
            if combined_type == 'l':
                h_fusion = self.cross_modal_fusion(h_l_quality)
            if combined_type == 'a':
                h_fusion = self.cross_modal_fusion(h_a_quality)
            if combined_type == 'v':
                h_fusion = self.cross_modal_fusion(h_v_quality)

        if type(h_fusion) == tuple:
            h_fusion = h_fusion[0]
        last_h_fusion = h_fusion[-1]  # Take the last output for prediction

        # A residual block
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_h_fusion)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_h_fusion

        output = self.out_layer(last_hs_proj)
        # print('output size:', output.size())

        return output, last_h_fusion, h_list
