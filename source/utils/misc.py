"""
This file contains function and data structure for general purpose
"""
import argparse
from source.utils.register import Registers
import re
from typing import List, Tuple
import functools
from pytorch_lightning import Trainer
import torch
import pathlib
import json

# ------------ GLOBAL VAR -------------------

MODEL_FILES_ROOT = pathlib.Path(__file__).parents[2] / "model_files"
proxy = { 'http': 'http://192.168.0.100:3128', 'https': 'http://192.168.0.100:3128' }


# ------------ HELPERS ----------------------

def switch_gpu_trainer(args, callbacks) :
    gpus = [i for i in range(torch.cuda.device_count())]
    print(f"+ Switching GPUS : {gpus} available")
    if gpus and len(gpus) > 1:
        args.gpus = str(gpus[0])

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)
    return trainer

def solve_metric(repr : str) -> Tuple[str, str, str] :
    """
    Convert a string representation into a double of string
    Check if a given metric follow the right format : [Name]|[Metric]
    if so check if the associated metric is available
    :param repr: representation of a given double name/metric
    :return: double representating the name of the metric and its associated metric
    """
    if re.match("[A-z]*\-[A-z]*\-Aa-z]*", repr):
        name, metric , _, set_name = repr.split("-")[2]
        repr = f"{name}-{metric}"
    else :
        set_name = "all"
    if re.match("[A-z]*-[A-z]*", repr) :
        name, metric =  repr.split("-")
        if metric not in Registers["METRICS"].keys() :
            available = Registers['METRICS'].keys()
            raise argparse.ArgumentTypeError(f"Unknown Metric {metric}. Available metrics "
                                             f"{available}")
        return name, metric, set_name
    else :
        raise argparse.ArgumentTypeError(f"Metrics wrong format {repr}. The pattern is [Name]-[Metric](-[Set])")

def create_or_append_metrics(metric : dict, path : pathlib) -> None :
    """
    Look for a specific json file, if found open and add the dictionary to the
    list otherwise create it
    :param metric:
    :param path:
    :return:
    """

    if not path.exists() :
        with path.open("w") as file :
            json.dump([metric], file, indent=4)
    else :
        with path.open("r") as file:
            data = json.load(file)

        data.append(metric)

        with path.open("w") as file :
            json.dump(data, file, indent=4)

tmd = 'TASKS|MODELS|DATASETS|LOSSES|CALLBACKS|METRICS|ALL'
def list_high_level_args(l : List[str]) -> None :
    """
    Pretty print the list of high levels arguments
    :param list: selected list to print out
    """

    if 'ALL' in l :
        l = tmd.split('|')[:-1]

    for args in l :

        available = Registers[args].keys()

        print(f'\nAvailable {args} are:\n',
              ''.join([f"\t{e}\n" for e in available]),
              end='\n\n\n')

def merge_namespaces(*namespaces, priority='first'):
    """
    Merge several namespaces together
    :param namespaces: list o²f namespaces
    :param priority: if namespaces share common keys the value 'last' allow
    overriding
    :return: Namespace
    """
    assert priority in {'first', 'last'}

    def merge_two_namespaces(n1, n2):
        ret = {**vars(n1)}
        for key, value in vars(n2).items():
            if key not in ret or priority == 'last':
                ret[key] = value
        return argparse.Namespace(**ret)

    return functools.reduce(merge_two_namespaces, namespaces)

def remove_path(path : pathlib.Path) :
    """
    Remove the provided file/directory even if its not empty.
    """
    if path.is_file() or path.is_symlink() :
        path.unlink()
        return
    for p in path.iterdir() :
        remove_path(p)
    path.rmdir()