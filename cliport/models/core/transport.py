import numpy as np
import cliport.models as models
from cliport.utils import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transport(nn.Module):

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        """Transport (a.k.a Place) module."""
        super().__init__()

        self.iters = 0
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations # -- # TODO
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        model = models.names[stream_one_fcn]
        self.key_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device, preprocess=utils.preprocess)
        self.query_resnet = model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, preprocess=utils.preprocess)
        print(f"Transport FCN: {stream_one_fcn}")

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1)  # spowers, slow: , padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape)# [1:])
        return output

    def transport(self, in_tensor, crop):
        logits = self.key_resnet(in_tensor)
        kernel = self.query_resnet(crop)
        return logits, kernel

    def forward(self, inp_img, p, softmax=True):
        """Forward pass."""
        padding = np.array([[0, 0]])
        padding = np.concatenate((padding, self.padding))
        img_unprocessed = np.pad(inp_img, padding, mode='constant')
        input_data = img_unprocessed

        if len(input_data.shape) == 3:
            in_shape = (1,) + input_data.shape
        else:
            in_shape = input_data.shape

        input_data = input_data.reshape(in_shape) # [B W H D]
        #device = self.key_stream_one._modules['conv1']._modules['0'].weight.device
        device = self.query_resnet.layer1._modules['0'].conv1.weight.device
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=device)

        # Rotation pivot.
        pv = np.array(p)[:, :] + self.pad_size
        if len(pv.shape) == 1:
            pv_shape = (1,) + pv.shape
            pv = pv.reshape(pv_shape)

        # Crop before network (default from Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2) # [B D W H]

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1, 1)  # R B C H W
        crop = self.rotator(crop, pivot=pv)
        crop = torch.stack(crop, dim=0)
        all_correlated_kernels = []

        for batch_id in range(pv.shape[0]):
            batch_crop = crop[:, batch_id, :, pv[batch_id, 0]-hcrop:pv[batch_id, 0]+hcrop, pv[batch_id, 1]-hcrop:pv[batch_id, 1]+hcrop]

            logits, kernel = self.transport(in_tensor[batch_id].unsqueeze(0), batch_crop)

            correlated_kernel = self.correlate(logits, kernel, softmax)  # 1, R, H, W
            #correlated_kernel = correlated_kernel.reshape(self.n_rotations, *correlated_kernel.shape[2:])  # R B C H W  # TODO: might not be right, rough.
            all_correlated_kernels.append(correlated_kernel)

        all_correlated_kernels = torch.cat(all_correlated_kernels, dim=0)

        # TODO(Mohit): Crop after network. Broken for now.
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)

        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        # kernel = crop[:, :, p[0]:(p[0] + self.crop_size), p[1]:(p[1] + self.crop_size)]

        return all_correlated_kernels.permute(0, 2, 3, 1)  # B H W R

