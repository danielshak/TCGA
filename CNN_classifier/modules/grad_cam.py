#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2

def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))

class _BaseWrapper(object):
    """
    Please modify forward() and backward() depending on your task.
    """

    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, idx):
        one_hot = torch.zeros((1, self.logits.size()[-1])).float()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)[0]
        return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)

    def backward(self, idx):
        """
        Class-specific backpropagation

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. self.logits[:, idx].backward(retain_graph=True)
        3. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = (self._encode_one_hot(idx)).double()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super().forward(self.image)

    def generate(self):
        gradient = self.image.grad.cpu().clone().numpy()
        self.image.grad.zero_()
        return gradient.transpose(0, 2, 3, 1)[0]


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=[]):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers

        def forward_hook(module, input, output):
            # Save featuremaps
            self.fmap_pool[id(module)] = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # Save the gradients correspond to the featuremaps
            self.grad_pool[id(module)] = grad_out[0].detach()

        # If any candidates are not specified, the hook is registered to all the layers.
        for module in self.model.named_modules():
            if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
                self.handlers.append(module[1].register_forward_hook(forward_hook))
                self.handlers.append(module[1].register_backward_hook(backward_hook))

    def _find(self, pool, target_layer):
        for key, value in pool.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError("Invalid layer name: {}".format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
#         print("shape", fmaps.shape)
#         print("fmaps", fmaps)
        
        grads = self._find(self.grad_pool, target_layer)
#         print("shape", grads.shape)
#         print("grads", grads)
        
        weights = self._compute_grad_weights(grads)
#         print("shape", weights.shape)
#         print("weights", weights)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
#         print("shape", gcam.shape)
#         print("gcam", gcam)
        
        gcam = torch.clamp(gcam, min=0.0)
#         print("shape", gcam.shape)
#         print("gcam", gcam)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.cpu().numpy()#, fmaps, grads, weights

