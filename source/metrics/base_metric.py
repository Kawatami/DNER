from __future__ import annotations
import torch
import torchmetrics
from typing import Any, Type
from argparse import ArgumentParser, Namespace

class BaseMetric(torchmetrics.Metric) :

    def __init__(self, dist_sync_on_step=False, log_name : str = None, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.log_name = log_name

    def update(self, batch : dict) -> None :
        raise NotImplementedError

    def compute(self) -> Any:
        raise NotImplementedError

    @staticmethod
    def add_metric_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        return parent_parser

    @classmethod
    def from_args(cls, args : Namespace) -> Type[BaseMetric] :
        """
        Build metrics from argument Namespace
        """
        return cls(**vars(args))