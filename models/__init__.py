# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib
from .simsiam import SimSiam
from .barlowtwins import BarlowTwins


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('../models')
            if not model.find('__') > -1 and 'py' in model]

names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args, backbone, loss, transform, device='cuda:0'):
    # args.model_name 是 simsiam, barlowtwins, None
    if not args.model_name is None:
        if args.model_name == 'simsiam':
            backbone = SimSiam(backbone).to(device)
            if args.model.proj_layers is not None:
                backbone.projector.set_layers(args.model.proj_layers)
        elif args.model_name == 'barlowtwins':
            backbone = BarlowTwins(backbone, device).to(
                device)
            if args.model.proj_layers is not None:
                backbone.projector.set_layers(args.model.proj_layers)

    return names[args.model](backbone, loss, args, transform)
