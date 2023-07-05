#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ***********************************************
#      Filename: wrapper/pruner/weight_mask.py
#        Author: jiff
#         Email: chuxiong.zhang@moffett.ai
#   Description: --
#        Create: 2023-03-10 16:37:19
# Last Modified: Year-month-day
# ***********************************************

import torch

from typing import Any, TypeVar

from torch import nn
from torch.nn import Module
from torch.nn.modules import conv
from torch.nn.parameter import Parameter, UninitializedParameter


MASK_COLLECTOR = dict()
MP_NAME = "_m_name"


class WeightMask:
    name: str

    def __init__(self, name: str="weight") -> None:
        self.name = name

    # TODO Make return type more specific
    def compute_weight(self, module: Module) -> Any:
        d = getattr(module, self.name + '_d')
        m = getattr(module, self.name + '_m')

        return d * m

    @staticmethod
    def apply(module, name: str) -> 'WeightMask':
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightMask) and hook.name == name:
                raise RuntimeError("Cannot register two weight_mask hooks on "
                                   "the same parameter {}".format(name))

        fn = WeightMask(name)

        if not hasattr(module, name):
            return

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                'The module passed to `WeightMask` can\'t have uninitialized parameters. '
                'Make sure to run a dummy forwarding process before applying weight mask'
            )
        # remove w from parameter list
        del module._parameters[name]

        # add d and m as new parameters and express w as d * m
        w_data, w_mask = weight, torch.ones_like(weight)
        module.register_parameter(name + '_d', w_data)
        module.register_buffer(name + '_m', w_mask)
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        # record the mask information
        if hasattr(module, MP_NAME):
            MASK_COLLECTOR[getattr(module, MP_NAME)] = w_mask

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_d']
        # del module._parameters[self.name + '_m']
        del module._buffers[self.name + '_m']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar('T_module', bound=Module)


def need_mask(module: T_module):
    return isinstance(module, (nn.Linear, conv._ConvNd))


def weight_mask(module: T_module, name: str = 'weight') -> T_module:
    r"""Applies weight mask to a parameter in the given module for pruning.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Returns:
        The original module with the weight mask hook

    Example::

        >>> m = weight_mask(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_d.size()
        torch.Size([40, 20])
        >>> m.weight_m.size()
        torch.Size([40, 20])

    """

    if need_mask(module):
        WeightMask.apply(module, name)

    return module


def remove_weight_mask(module: T_module, name: str = 'weight') -> T_module:
    r"""Removes the weight mask reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_mask(nn.Linear(20, 40))
        >>> remove_weight_mask(m)
    """
    if need_mask(module):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightMask) and hook.name == name:
                hook.remove(module)
                del module._forward_pre_hooks[k]

                # remove the mask information
                if hasattr(module, MP_NAME) and getattr(module, MP_NAME) in MASK_COLLECTOR:
                    MASK_COLLECTOR.pop(getattr(module, MP_NAME))

                return module

        raise ValueError("weight_mask of '{}' not found in {}".format(name, module))


def name_model(module):
    for name, submodule in module.named_modules():
        setattr(submodule, MP_NAME, name)


def remove_name_model(module):
    for _, submodule in module.named_modules():
        if hasattr(submodule, MP_NAME):
            delattr(submodule, MP_NAME)


def feed_mask(mask_dict: dict):
    for name, mask in mask_dict.items():
        if name in MASK_COLLECTOR:
            MASK_COLLECTOR[name].data = mask
        else:
            print(f"Found redundent mask for `{name}` which is not required.")


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.bloom.modeling_bloom import BloomForCausalLM

    model_cache_dir = '/ssd1/models/bloom'

    # model definition
    pretrained_id = "bigscience/bloom-560m"
    config = AutoConfig.from_pretrained(pretrained_id, cache_dir=model_cache_dir)
    config.n_layer = 4
    model = BloomForCausalLM(config)

    # prepare model input
    tokenizer = AutoTokenizer.from_pretrained(pretrained_id, use_fast=True, cache_dir=model_cache_dir)
    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]
    dummy_inputs = tokenizer(batch_sentences, padding=True, return_tensors='pt')

    print("#" * 60)
    print(f"Model structure:\n{model}")

    # 1. name module
    name_model(model)

    # 2. apply mask before training
    model.apply(weight_mask)
    print("#" * 60)

    print(f"Adding mask to weights..")
    MASK_COLLECTOR_INFO = {name: param.shape for name, param in MASK_COLLECTOR.items()}
    print(f"Mask collector: {MASK_COLLECTOR_INFO}")

    # feed data to mask
    mask_dict = {
        'transformer.h.0.self_attention.query_key_value': torch.zeros_like(
            model.transformer.h[0].self_attention.query_key_value.weight
        )
    }
    feed_mask(mask_dict)

    # training
    y = model(**dummy_inputs)

    # 3. remove mask after training
    model.apply(remove_weight_mask)
    print("#" * 60)

    print(f"Removing mask to weight..")
    MASK_COLLECTOR_INFO = {name: param.shape for name, param in MASK_COLLECTOR.items()}
    print(f"Mask collector: {MASK_COLLECTOR_INFO}")

    y = model(**dummy_inputs)

    # 4. remove name from module
    remove_name_model(model)
