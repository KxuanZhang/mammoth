# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            print(self.buffer.examples.shape)
            s_t = time.time()
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            e_t = time.time()
            print('buffer search time', e_t-s_t)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        s_t = time.time()
        outputs = self.net(inputs)
        e_t = time.time()
        print('model inference time', e_t - s_t)

        s_t = time.time()
        loss = self.loss(outputs, labels.type(torch.LongTensor).to(self.device))
        e_t = time.time()
        print('loss calculate time', e_t - s_t)

        s_t = time.time()
        loss.backward()
        self.opt.step()
        e_t = time.time()
        print('反向传播和参数更新time', e_t - s_t)

        s_t = time.time()
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        e_t = time.time()
        print('buffer 增加的time', e_t - s_t)
        # import  os
        # os.pause()

        return loss.item()
