# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Authors: Yossi Adi (adiyoss) and Alexandre Défossez (adefossez)

import json
import logging
import math
from pathlib import Path
import os
import re

import librosa
import numpy as np
import torch
import torch.utils.data as data

from .preprocess import preprocess_one_dir
from .audio import Audioset

logger = logging.getLogger(__name__)


def sort(infos): return sorted(
    infos, key=lambda info: int(info[1]), reverse=True)


class Trainset:
    def __init__(self, json_dir, sample_rate=16000, segment=4.0, stride=1.0, pad=True):
        mix_json = os.path.join(json_dir, 'mix.json')
        s_jsons = list()
        s_infos = list()
        sets_re = re.compile(r's[0-9].json')
        for s in os.listdir(json_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(json_dir, s))

        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, 'r') as f:
                s_infos.append(json.load(f))

        length = int(sample_rate * segment)
        stride = int(sample_rate * stride)

        kw = {'length': length, 'stride': stride, 'pad': pad}
        self.mix_set = Audioset(sort(mix_infos), True, **kw)

        self.sets = list()
        for s_info in s_infos:
            self.sets.append(Audioset(sort(s_info), False, **kw))

        # verify all sets has the same size
        for s in self.sets:
            print (f'{len(s)} and {len(self.mix_set)}')
            assert len(s) == len(self.mix_set)

    def __getitem__(self, index):
        mix_sig = self.mix_set[index]
        tgt_sig = [self.sets[i][index] for i in range(len(self.sets))]
            
        return torch.stack(mix_sig), torch.LongTensor([mix_sig[0].shape[0]]), torch.stack(tgt_sig)

    def __len__(self):
        return len(self.mix_set)


class Validset:
    """
    load entire wav.
    """

    def __init__(self, json_dir):
        mix_json = os.path.join(json_dir, 'mix.json')
        s_jsons = list()
        s_infos = list()
        sets_re = re.compile(r's[0-9].json')
        for s in os.listdir(json_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(json_dir, s))
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, 'r') as f:
                s_infos.append(json.load(f))
        self.mix_set = Audioset(sort(mix_infos), True)
        self.sets = list()
        for s_info in s_infos:
            self.sets.append(Audioset(sort(s_info),False))
        for s in self.sets:
            print (f'{len(s)} and {len(self.mix_set)}')
            assert len(s) == len(self.mix_set)

    def __getitem__(self, index):
        mix_sig = self.mix_set[index]
        tgt_sig = [self.sets[i][index] for i in range(len(self.sets))]
        return torch.stack(mix_sig), torch.LongTensor([mix_sig[0].shape[0]]), torch.stack(tgt_sig)

    def __len__(self):
        return len(self.mix_set)


# The following piece of code was adapted from https://github.com/kaituoxu/Conv-TasNet
# released under the MIT License.
# Author: Kaituo XU
# Created on 2018/12
class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix[0].shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    # mixtures_pad = pad_list([torch.from_numpy(mix).float()
    #                          for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)

    #mixtures_pad = mixtures_pad.permute((0, 2, 1)).contiguous()
    mixtures_pad = torch.stack(mixtures[0])
    mixtures_pad = mixtures_pad.reshape(1,len(mixtures[0]),ilens[0])

    return mixtures_pad, ilens, filenames


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        filename = mix_path[0][:-8] + ".wav"
        # read wav file
        mix_files = mix_path
        mix_array = []
        for file_path in mix_files:
            mix, _ = librosa.load(file_path, sr=sample_rate)
            mix_array.append(mix)   

        min_size= np.min([len(mix) for mix in mix_array])
        mix_array = [torch.Tensor(mix[:min_size]) for mix in mix_array]

        mixtures.append(mix_array)
        filenames.append(filename)
        
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad
