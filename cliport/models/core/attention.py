"""Attention module."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.models as models
from cliport.utils import utils


class Attention(nn.Module):
    """Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__()
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        self.attn_stream = models.names[stream_one_fcn](self.in_shape, 1, self.cfg, self.device, self.preprocess)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x):
        return self.attn_stream(x)

    def forward(self, inp_img, softmax=True):
        """Forward pass."""
        padding = np.array([[0, 0]])
        padding = np.concatenate((padding, self.padding))
        in_data = np.pad(inp_img, padding, mode='constant')

        if len(in_data.shape) == 3:
            in_shape = (1,) + in_data.shape
        else:
            in_shape = in_data.shape

        in_data = in_data.reshape(in_shape)

        #device = self.attn_stream.layers._modules['0'].weight.device
        device = self.attn_stream.layer1._modules['0'].conv1.weight.device
        #device = self.attn_stream_one._modules['conv1']._modules['0'].weight.device
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=device) # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1, 1)  # [R B 6 W H]
        in_tens = self.rotator(in_tens, pivot=np.expand_dims(pv, 0))

        # Forward pass.  # TODO: why is this a for loop instead of a batch?
        logits = []
        for x in in_tens:
            lgts = self.attend(x)
            logits.append(lgts)
        logits = torch.stack(logits, dim=0)  # OG: cat. Stacking to preserve dims for reversal

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=np.expand_dims(pv, 0))
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[1:3]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]  # TODO: check...

        logits = logits.permute(0, 2, 3, 1)  # [B W H 1] -- TODO: spowers, last dim is rotation, I think. No wait, that got cat'd away.
        output = logits.reshape(logits.shape[0], np.prod(logits.shape[1:]))
        if softmax:
            output = F.softmax(output, dim=-1)

        output = output.reshape(logits.shape)
        return output