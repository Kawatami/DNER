from __future__ import annotations
import torch
from typing import List, Optional, Iterable
import itertools
from collections import UserDict

class CustomDict(UserDict) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.lock_batch = False

    def lock(self) :
        self.lock_batch = True

    def unlock(self):
        self.lock_batch = False

    def __setitem__(self, key, value):
        if key in self and self.lock_batch :
            raise ValueError(f"Do not modify an existing key : {key}")
        else :
            super().__setitem__(key, value)


class BatchDict(CustomDict) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

    def transfer_to_CPU(self) -> dict:
        res = {}

        for key, value in self.items() :
            if isinstance(value, torch.Tensor):
                value = value.to('cpu')
            res[key] = value

        return res

    def get_dict(self) :
        return { k : v for k, v in self.items() }

    def convert_to_tensors(self, list_keys) :
        for key in list_keys :
            if key not in self :
                raise KeyError(f"Key not found {key}. Available keys {self.keys()}")
            else :
                self[key] = torch.tensor(self[key])

    def __repr__(self) :
        res = "==== BATCH ====\n"
        for key, value in self.items() :
            res += f"\t+ {key} : {value}\n"

        return res

class DataSample(object) :

    def __init__(self, gpu : Optional[dict] = None,
                       cpu : Optional[dict] = None,
                       **kwargs) :
        super().__init__(**kwargs)
        self.cpu = CustomDict() if cpu is None else CustomDict(**cpu)
        self.gpu = CustomDict() if gpu is None else CustomDict(**gpu)

    def items(self) -> Iterable :
        return itertools.chain(self.cpu.items(), self.gpu.items())

    def __getitem__(self, item) :
        if item in self.cpu :
            return self.cpu[item]
        elif item in self.gpu :
            return self.gpu[item]
        else :
            raise KeyError(f"{item} key not found"
                           f"available cpu keys : {self.cpu.keys()}"
                           f"avalaible gpu keys : {self.gpu.keys()}")

    def get_tensor_keys(self) -> dict.keys :
        return self.gpu.keys()

    @staticmethod
    def merge_samples(samples : List[DataSample]) -> dict :

        batch = BatchDict()

        # get tensor keys
        tensor_keys = list({ key for sample in samples for key in sample.get_tensor_keys() })

        # for all samples
        for sample in samples :
            for key, value in sample.items() :
                batch_value = batch.get(key, [])
                batch_value.append(value)
                batch[key] = batch_value

        # convert to tensor
        batch.convert_to_tensors(tensor_keys)

        batch.lock()

        return batch.get_dict()

    def __repr__(self) :
        res = "+ CPU data : \n"
        for key, value in self.cpu.items() :
            res += f"\t o {key} : {value} {type(value)}\n"

        res += "+ GPU data : \n"
        for key, value in self.gpu.items() :
            res += f"\t o {key} : {value} {type(value)}\n"

        return res
