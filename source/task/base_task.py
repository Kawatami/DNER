import argparse
from argparse import ArgumentParser, Namespace
from typing import List, Any, Optional
from source.utils.register import Registers

import pytorch_lightning as pl
import torch
from torch.nn import ModuleList, ModuleDict
import copy
import math

class BaseTask(pl.LightningModule):
    def __init__(self,
                 train_set_names: Optional[List[str]] = None,
                 val_set_names: Optional[List[str]] = None,
                 test_set_names: Optional[List[str]] = None,
                 **kwargs):
        super().__init__()
        args = Namespace(**kwargs)

        # datasets names
        self.train_set_names : Optional[List[str]] = train_set_names
        self.val_set_names : Optional[List[str]] = val_set_names
        self.test_set_names : Optional[List[str]] = test_set_names

        # building model and loss
        self.model = self.build_model_from_args(args)

        self.loss, self._loss_name = self.build_loss_from_args(args)

        # storing metrics
        self.metrics = {
            'trainset': list(),
            'valset': list(),
            'testset': list()
        }

        self.build_metrics(args)

        #print("=====================")
        #print(self.metrics)
        #self.save_hyperparameters()


    def get_model_sample_creation_method(self) :
        return self.model.create_samples

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(**vars(args))

    @staticmethod
    def build_model_from_args(args):
        return Registers['MODELS'][args.model].from_args(args)

    @staticmethod
    def build_loss_from_args(args):
        loss = Registers['LOSSES'][args.loss].from_args(args)
        return loss, loss.name

    @staticmethod
    def build_metrics_from_args(args):
        metrics = []
        for name, metric_name, set_name in args.metrics :
            print(name, metric_name, set_name)
            args.log_name = name
            metric = Registers["METRICS"][metric_name].from_args(args)
            metrics.append((metric, set_name))
        return metrics

    def model_gradient_norm(self):
        accumulator = 0
        for name, param in self.model.named_parameters() :

            if param.grad is not None :
                accumulator += param.grad.norm(2) ** 2

        return accumulator ** (1/2)

    def build_metrics(self, args : argparse.Namespace) :

        def None2one(set) :
            return len(set) if set is not None else 1

        num_sets = [
            None2one(self.train_set_names),
            None2one(self.val_set_names),
            None2one(self.test_set_names)
        ]

        # storing metrics
        self.metrics = {
            'trainset': list(),
            'valset': list(),
            'testset': list()
        }

        metrics = self.build_metrics_from_args(args)

        for dataset, num_set in zip(self.metrics.keys(), num_sets) :
            for i in range(num_set) :
                metric_list = []
                for metric, set_name in metrics:
                    if dataset == set_name or set_name == "all":
                        metric_list.append(copy.deepcopy(metric))
                self.metrics[dataset].append(metric_list)


    def step(self, batch : dict, subset : str, name : str = None, metrics : Optional[List] = None) :

        # processing model output
        model_output = self.model(batch)  # prediction
        loss = self.loss(model_output, batch)  # loss compute

        if isinstance(loss, tuple) :
            loss, loss_sample = loss
        else :
            loss_sample = None

        log = {}
        name = name if name is not None else subset
        # handling multi task loss case
        if isinstance(loss, tuple):
            log.update({f"{key}/{name}": value for key, value in loss[1].items()})
            loss = loss[0]

        # log loss
        log.update({f'{self._loss_name}/{name}': loss})

        # log mode gradient
        if name == "trainset" :
            log.update({f"model_gradient/{name}" : self.model_gradient_norm()})

        # update metrics
        if metrics is not None :
            for metric in metrics :
                metric.update(batch)

        return loss, log, model_output, loss_sample

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        name = self._get_dataset_name(self.train_set_names, dataloader_idx)
        idx = 0 if dataloader_idx is None else dataloader_idx
        metrics = self.metrics['trainset'][idx]
        loss, log, _, loss_sample = self.step(batch, 'trainset', name=name, metrics=metrics)
        self.log_dict(log, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        name = self._get_dataset_name(self.val_set_names, dataloader_idx)
        idx = 0 if dataloader_idx is None else dataloader_idx

        metrics = self.metrics['valset'][idx]
        loss, log, outputs, loss_sample = self.step(batch, 'valset', name=name, metrics=metrics)
        self.log_dict(log, prog_bar=True)

        return { "loss" : loss, "loss_sample" : loss_sample , "output" : outputs }

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        # switching to test mode
        if hasattr(self.model, "test_mode") :
            self.model.test_mode = True

        name = self._get_dataset_name(self.test_set_names, dataloader_idx)

        metrics = self.metrics['testset'][dataloader_idx]
        loss, log, outputs, loss_sample = self.step(batch, 'testset', name=name, metrics=metrics)
        self.log_dict(log, prog_bar=True)

        # free memory from GPU
        outputs = self.transfer_to_cpu(outputs)

        return outputs

    def transfer_to_cpu(self, sample : dict) -> dict :
        """
        Transfer tensor to CPU.
        :param sample: batch as dictionary
        """
        res = {}
        for key, value in sample.items() :
            if isinstance(value, torch.Tensor) :
                res[key] = value.to("cpu")

        return res

    def _get_dataset_name(self, names, dataloader_idx) :

        if names is None or dataloader_idx is None :
            return None

        if dataloader_idx >= len(names) :
            return None

        return names[dataloader_idx]

    def epoch_end(self, subset : str, names_dataLoader : Optional[List[str]]) -> None :
        """
        Log the metrics. Metrics can either return the result or a list of results of the form [(name, result)]
        :param subset: subset to select metrics
        """

        # if subset admit metric
        if self.metrics[subset] :
            log = {}

            # iterate over metrics
            for set_idx, metric_list in enumerate(self.metrics[subset]) :

                name_dataLoader = n if (n := self._get_dataset_name(names_dataLoader, set_idx)) else subset

                for metric in metric_list :

                    # compute results
                    metric_result = metric.compute()

                    # if metric result is a list
                    if isinstance(metric_result, list) :
                        # log individually
                        for name, result in metric_result :
                            log[f'{name}/{name_dataLoader}'] = result
                    else :
                        log[f'{metric.name}/{name_dataLoader}'] = metric_result

            self.log_dict(log, prog_bar=True)


    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.epoch_end('trainset', self.train_set_names)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.epoch_end('valset', self.val_set_names)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.epoch_end('testset', self.test_set_names)

    def get_progress_bar_dict(self):
        return dict()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)