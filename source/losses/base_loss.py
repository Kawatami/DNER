from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register, Registers
import torch
from typing import Tuple, Type

class BaseLoss:
    """
    Base class for Loss object
    """
    def __init__(self, log_name : str = None, **kwargs) :
        """
        Base class constructor
        :param log_name: name to use in the case of metric usage
        """
        self.log_name = log_name

    @staticmethod
    def _prep_inputs(model_outputs : dict, batch : dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the model output and labels to be processed by a loss object
        :param model_outputs: model output
        :param batch: batch object
        :return: Associated tensors
        """
        prd, tgt = model_outputs['prediction'], batch['labels'].float()
        prd, tgt = prd.view(-1), tgt.view(-1)
        return prd, tgt

    @staticmethod
    def add_loss_specific_args(parent_parser : ArgumentParser) -> ArgumentParser:
        return parent_parser

    @classmethod
    def from_args(cls, args : Namespace) -> Type[BaseLoss]:
        """
        Build Loss object from Namespace issued by main parser
        :param args:
        :return:
        """

        return cls(**vars(args))

    @property
    def abbrv(self):
        return self.__class__.__name__

@register('LOSSES')
class MultiTaskLoss(torch.nn.Module):
    """
    Base class for Multi task learning.

    ARGS :
        losses Union[BaseLoss] : list of losses
        weights Union[float] : one weight per loss, they sum to one.
        inputs Union[str] : list of prediction keys on which to apply the losses
        outputs Union[str] : list of gold standard keys for each loss
    """

    _names = ['MultiTask']

    @property
    def name(self):
        return 'MT'

    def __init__(self, losses, weights, input_keys, target_keys, *args, **kwargs):
        super().__init__()

        # input sanity check
        if len(weights) == 1 and len(losses) == 2:
            weights.append(1 - weights[0])
        else:
            assert len(weights) == len(losses), f'{len(weights)} weights specified, but you have {len(losses)} losses!'
        #assert sum(weights) == 1

        unknown_losses = [loss for loss in losses if loss not in Registers['LOSSES']]
        if len(unknown_losses):
            raise ValueError(f'Unknown loss: {unknown_losses}')

        # defining losses
        # TODO: deal with args in losses (in a future version)
        self.losses = torch.nn.ModuleList([
            Registers['LOSSES'][loss]() for loss in losses
        ])
        self.weights = weights

        # setting dict names
        self.input_keys = input_keys
        self.target_keys = target_keys

        #
        self.last_state = {}

    @classmethod
    def from_args(cls, args):
        return cls(losses=args.losses,
                   weights=args.loss_weights,
                   input_keys=args.loss_input_keys,
                   target_keys=args.loss_target_keys)

    @staticmethod
    def add_loss_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Multi-task loss')

        group.add_argument("--losses", type=str, nargs='+',
                           default=['LogCosh', 'LogCosh'])
        group.add_argument("--loss_weights", type=float, nargs='+',
                           default=[0.5, 0.5])
        group.add_argument("--loss_input_keys", type=str, nargs='+',
                           default=['prediction', 'ae_prediction'])
        group.add_argument("--loss_target_keys", type=str, nargs='+',
                           default=['tgt', 'values'])
        return parser

    def update_weights(self, weights):
        self.weights = weights

    def _prep_inputs(self, model_outputs, loss_input):
        """
        Bottle inputs the correct way to be processed by BaseLoss
        """
        return [
            [{'prediction': model_outputs[input_key]},
             {'tgt': loss_input[output_key]}]
            for input_key, output_key in zip(self.input_keys, self.target_keys)
        ]

    def forward(self, model_output, loss_input):
        iterable = zip(
            self.losses,
            self.weights,
            self._prep_inputs(model_output, loss_input),
            self.input_keys,
            self.target_keys
        )

        individual_logs = {}

        loss = 0
        for loss_func, weight, (prd, tgt), in_key, tgt_key in iterable:
            current_loss = loss_func(prd, tgt)
            loss += weight * loss_func(prd, tgt)
            individual_logs.update({
                f"{loss_func.abbrv}_{in_key}_{tgt_key}": current_loss
            })
        return loss, individual_logs