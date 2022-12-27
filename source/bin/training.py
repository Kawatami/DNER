from argparse import Namespace
from source.utils.register import Registers
from source.callbacks.factory import CallbacksFactory
from source.utils.misc import switch_gpu_trainer, remove_path
from pytorch_lightning import Trainer
import pathlib
import warnings
from source.data.tokenizer.tokenizer import TokenizerLoader
import logging

def train(args : Namespace) -> int :
    """
    Initialize model, task and dataset in order to train a model
    :param args: Namespace with all the needed arguments
    :return: 0 if successful 1 otherwise
    """


    # setting logging to display output
    logging.basicConfig(level=logging.DEBUG)

    # Init Tokenizer loader
    TokenizerLoader(**vars(args))

    # Init task and dataset
    task    = Registers["TASKS"][args.task].from_args(args)
    dataset = Registers["DATASETS"][args.dataset].from_args(args)

    # registering sample creation method
    dataset.create_sample_method = task.get_model_sample_creation_method()

    # print out model
    if args.verbose >= 2 :
        print(task.model)

    # building callbacks
    callback_factory = CallbacksFactory(args)
    callbacks = callback_factory.build_callbacks()

    # Init trainer
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    try :

        # launching training
        trainer.fit(task, datamodule=dataset)

        if args.single_gpu_testing :
            trainer = switch_gpu_trainer(args, callbacks)
            trainer.test(model=task, test_dataloaders=dataset.test_dataloader())
        else :
            trainer.test(test_dataloaders=dataset.test_dataloader())

    finally :

        if args.remove_output_data :
            warnings.warn(f"Removing {args.collector_log_dir}")
            remove_path(args.collector_log_dir)

            if trainer.log_dir :
                warnings.warn(f"Removing {trainer.log_dir}")
                remove_path(pathlib.Path(trainer.log_dir))

    return 0


