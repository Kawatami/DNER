from __future__ import annotations

import logging

import torch
from typing import Type, Any
from argparse import ArgumentParser

class DNETBase(torch.nn.Module) :
    """
    Baseline model for DNET task
    """

    def __init__(self,
                 context_encoder : Type[torch.nn.Module],
                 span_encoder : Type[torch.nn.Module],
                 classifier : Type[torch.nn.Module],
                 last_activation : str = "sigmoid") :
        """
        Base constructor
        :param context_encoder: Take input sequence and produce embedded token
        sequence
        :param span_encoder: extract span and produce vector representation of
        shape [num_spans, span_dim]
        :param classifier: classifier for span should produce vector of shape
        [num_span, 1] with probabilities coeficients
        """
        super().__init__()
        self.context_encoder = context_encoder
        self.span_encoder = span_encoder
        self.classifier =  classifier
        self.last_activation = last_activation

    def freeze_model(self) :
        logging.info(f"Freezing model")
        for param in self.parameters() :
                param.requires_grad = False

    def unfreeze_model(self) :
        logging.info(f"Unfreezing model")
        for param in self.parameters() :
                param.requires_grad = True

    def unfreeze_layers(self, names) :
        logging.info(f"Unfreezing model layer : {names}")
        for name, param in self.named_parameters() :
            if name.split(".")[-1] in names :
                param.requires_grad = True

    def freeze_layers(self, names) :
        logging.info(f"Unfreezing model layer : {names}")
        for name, param in self.named_parameters() :
            if name.split(".")[-1] in names :
                param.requires_grad = False

    def forward(self, batch : dict) :
        """
        inference method
        :param batch: batch of data
        :return: batch of data updated with intermediary states
        """
       # print(batch)

        # context encoding
        batch = self.context_encoder(batch)

        # span encoding
        batch = self.span_encoder(batch)

        # classification
        batch = self.classifier(batch)

        if self.last_activation == "sigmoid" :
            #print(batch['prediction'].size())
            batch['prediction'] = torch.sigmoid(batch['prediction']).to(torch.float)
           # print(batch['prediction'].size())

            batch['prediction_label'] = (batch['prediction'] > 0.5).int()
           # print(batch['prediction_label'].size())


        else :
            batch['prediction'] = batch['prediction'].permute(0, 2, 1)
            batch['prediction_label'] = batch['prediction'].softmax(dim=1).argmax(dim=1)

        return batch

    @staticmethod
    def add_model_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        """
        Update main parser with model specific args
        :param parent_parser: main parser
        :return: main  parser updated
        """
        raise NotImplementedError

    @classmethod
    def from_args(cls, args) -> Type[DNETBase]:
        raise NotImplementedError

    def create_samples(self, raw_samples : Any, **kwargs) :
        raise NotImplementedError

