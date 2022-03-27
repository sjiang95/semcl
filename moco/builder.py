# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from info_nce import InfoNCE

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0,loss_mode=None):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        #InfoNCE
        self.loss_mode=loss_mode
        self.infonce=InfoNCE(temperature=T,reduction='mean',negative_mode='paired')

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            #set input dim of mlp
            dim1 = input_dim if l == 0 else mlp_dim
            #set output dim of mlp
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            #add mlp layer
            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:#if it is not the last mlp layer
                #add bachnorm
                mlp.append(nn.BatchNorm1d(dim2))
                #and relu
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: anchor images
            x2: ~anchor images
            m: moco momentum
        Output:
            loss
        """
        loss=None
        # compute features
        q_anchor0 = self.predictor(self.base_encoder(torch.squeeze(x1[:,0])))
        q_anchor1 = self.predictor(self.base_encoder(torch.squeeze(x1[:,1])))
        q_anchor2 = self.predictor(self.base_encoder(torch.squeeze(x1[:,2])))

        q_nanchor0 = self.predictor(self.base_encoder(torch.squeeze(x2[:,0])))
        q_nanchor1 = self.predictor(self.base_encoder(torch.squeeze(x2[:,1])))
        q_nanchor2 = self.predictor(self.base_encoder(torch.squeeze(x2[:,2])))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k_anchor0 = self.momentum_encoder(torch.squeeze(x1[:,0]))
            k_anchor1 = self.momentum_encoder(torch.squeeze(x1[:,1]))
            k_anchor2 = self.momentum_encoder(torch.squeeze(x1[:,2]))

            k_nanchor0 = self.momentum_encoder(torch.squeeze(x2[:,0]))
            k_nanchor1 = self.momentum_encoder(torch.squeeze(x2[:,1]))
            k_nanchor2 = self.momentum_encoder(torch.squeeze(x2[:,2]))

        # TODO: use all other samples in the same batch as negative samples if current method performs bad
        pos_keys_stack=torch.stack([k_anchor0,k_anchor1,k_anchor2],dim=1)
        neg_keys_stack=torch.stack([k_nanchor0,k_nanchor1,k_nanchor2],dim=1)

        if self.loss_mode=='L0' or self.loss_mode=='L':
            loss0=(self.infonce(q_anchor0,k_anchor1,negative_keys=neg_keys_stack)
                    +self.infonce(q_anchor1,k_anchor2,negative_keys=neg_keys_stack)
                    +self.infonce(q_anchor2,k_anchor0,negative_keys=neg_keys_stack)
                    +self.infonce(q_nanchor0,k_nanchor1,negative_keys=pos_keys_stack)
                    +self.infonce(q_nanchor1,k_nanchor2,negative_keys=pos_keys_stack)
                    +self.infonce(q_nanchor2,k_nanchor0,negative_keys=pos_keys_stack)
                )
            # loss0/=float(4.0)

        if self.loss_mode=='mocov3' or self.loss_mode=='L':
            loss_view0=self.contrastive_loss(q_anchor0,k_anchor1)+self.contrastive_loss(q_nanchor0,k_nanchor1)
            loss_view1=self.contrastive_loss(q_anchor1,k_anchor2)+self.contrastive_loss(q_nanchor1,k_nanchor2)
            loss_view2=self.contrastive_loss(q_anchor2,k_anchor0)+self.contrastive_loss(q_nanchor2,k_nanchor0)
            loss_mocov3=loss_view0+loss_view1+loss_view2
            # loss1/=float(4.0)
        if self.loss_mode=='L':
            return loss0+loss_mocov3
        elif self.loss_mode=='L0':
            return loss0
        elif self.loss_mode=='mocov3':
            return loss_mocov3

class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
