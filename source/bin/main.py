from argparse import ArgumentParser
from source.utils.register import Registers
from source.utils.misc import (
    solve_metric,
    tmd,
    list_high_level_args,
    merge_namespaces
)
from source.bin.training import train
from pytorch_lightning import Trainer
from source.callbacks.factory import default_callback, CallbacksFactory
from source.data.tokenizer.tokenizer import TokenizerLoader
import warnings
import pathlib

# setting proxy

import os

#proxy = 'http://hacienda:3128'

#os.environ['http_proxy'] = proxy
#os.environ['HTTP_PROXY'] = proxy
#os.environ['https_proxy'] = proxy
#os.environ['HTTPS_PROXY'] = proxy

#
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main() -> int :

    description = "RotoWire main training script. Only high level arguments are \
    showed. User still need to provide trainer, task, model, dataset and loss \
     args"

    high_level_parser = ArgumentParser(description = description)

    # helper function
    high_level_parser.add_argument('--list', type=str, nargs='+',
                                   default=list(),
                                   choices=tmd.split("|"),
                                   help=f'List available {tmd} and exit. ' \
                                        "Select 'all' to show all.")

    # helper function

    high_level_parser.add_argument("--collector_log_dir",
                                   type=pathlib.Path,
                default=pathlib.Path(__file__).parents[2] / "data" / "inference")

    high_level_parser.add_argument(
        "--remove_output_data", action="store_true", default=False,
        help="Remove the produced files automatically at the end of the testing phase. Intended to be used for debugging"
             "purposes"
    )

    high_level_parser.add_argument('--debug_test_set', default=False,
                                   action="store_true",
                                   help=f"Set the max steps to 1 with no callbacks"
                                        f".Used for debuging purpose")


    high_level_parser.add_argument('--single_gpu_testing', default=False,
                                   action="store_true",
                                   help=f"Using only one gpu during testing"
                                        f"Used when for callback datacollection")

    high_level_parser.add_argument('--verbose', type=int, default=1,
                                   choices=[0, 1, 2, 3],
                                   help=f'verbose mode ' \
                                        "0 (no print), 1 (training info), "
                                        "2 (+model info), 3 (everything)")

    high_level_parser.add_argument('--callback_verbose', type=bool, default=False,
                                   help=f'Enable verbose mode on all callback')

    # high level arguments
    high_level_parser.add_argument('--task', type=str, required=True,
                                   choices=Registers["TASKS"].keys())
    high_level_parser.add_argument('--model', type=str,required=True,
                                   choices=Registers["MODELS"].keys())
    high_level_parser.add_argument('--loss', type=str, required=True,
                                   choices=Registers["LOSSES"].keys())
    high_level_parser.add_argument('--metrics', type=solve_metric, default = [],
                                   nargs='+')
    high_level_parser.add_argument('--callbacks', type=str, nargs='+', default = [],
                                   choices=Registers["CALLBACKS"].keys())

    # tokenizer args
    high_level_parser = TokenizerLoader.add_tokenizer_loader_specific_args(high_level_parser)

    # parse high level argument
    high_level_args, remaining_args = high_level_parser.parse_known_args()

    # print out list and quitting
    if len(high_level_args.list) :
        list_high_level_args(high_level_args.list)
        return 0

    # low level training args parser
    parser = ArgumentParser(description='Internal parser.')
    parser = Trainer.add_argparse_args(parser)
    parser = Registers["TASKS"][high_level_args.task].add_task_specific_args(parser)
    parser = Registers["MODELS"][high_level_args.model].add_model_specific_args(parser)
    parser = Registers["LOSSES"][high_level_args.loss].add_loss_specific_args(parser)
    parser = CallbacksFactory.add_callbackFactory_specific_args(parser)

    for callback in high_level_args.callbacks + default_callback :
        parser = Registers["CALLBACKS"][callback].add_callback_specific_args(parser)

    for name, metric, _ in high_level_args.metrics :
        parser = Registers["METRICS"][metric].add_metric_specific_args(parser)

    training_args, remaining_args = parser.parse_known_args(remaining_args)

    # low level dataset args parser
    parser = Registers["DATASETS"][training_args.dataset].add_data_specific_args(parser)
    data_args = parser.parse_args(remaining_args)

    # merging all the namespaces produce by the different parcers
    args = merge_namespaces(high_level_args, training_args, data_args)

    # In case of debugging overriding param
    if args.debug_test_set :
        warnings.warn(f"\x1b[33;20m###### Test set DEBUG mode enable, max step set to 1, no default callbacks\x1b[0m")
        args.no_default_callback = True
        args.max_steps = 1

    if args.remove_output_data :
        warnings.warn(f"\x1b[31;20m###### remove_output_data enable, data and log files will automaticaly be removed at the end "
                      f"of the script\x1b[0m")

    return train(args)

if __name__ == "__main__" :
    main()
