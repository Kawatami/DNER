from __future__ import annotations
import torch
from typing import Type, Any, Optional
from argparse import ArgumentParser

class DNERBase(torch.nn.Module) :
    """
    Baseline model for rotowire task 2
    """

    def __init__(self,
                 context_encoder : Type[torch.nn.Module],
                 classifier : Type[torch.nn.Module],
                 word_pooling : Optional[Type[torch.nn.Module]]) :
        """
        Base constructor
        :param context_encoder: Take input sequence and produce embedded token
        sequence
        :param classifier: classifier for span should produce vector of shape
        [num_span, 1] with probabilities coefficients
        """
        super().__init__()
        self.context_encoder = context_encoder
        self.classifier =  classifier
        self.word_pooling = word_pooling

    def forward(self, batch : dict) :
        """
        inference method
        :param batch: batch of data
        :return: batch of data updated with intermediary states
        """
        # context encoding
        batch = self.context_encoder(batch)

        if self.word_pooling is not None :
            # pooling to word token
            batch = self.word_pooling(batch)

        # classification
        batch = self.classifier(batch)
        batch['prediction'] = batch['prediction'].permute(0, 2 ,1)
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
    def from_args(cls, args) -> Type[DNERBase]:
        raise NotImplementedError

    def create_samples(self, raw_samples : Any, **kwargs) :
        raise NotImplementedError

class RotoWIreTask2BaseModelSequential(torch.nn.Module) :
    """
    Baseline model for rotowire task 2
    """

    def __init__(self,
                 context_encoder : Type[torch.nn.Module],
                 entity_classifier : Type[torch.nn.Module],
                 type_classifier: Type[torch.nn.Module],
                 word_pooling : Type[torch.nn.Module]) :
        """
        Base constructor
        :param context_encoder: Take input sequence and produce embedded token
        sequence
        :param classifier: classifier for span should produce vector of shape
        [num_span, 1] with probabilities coeficients
        """
        super().__init__()
        self.context_encoder = context_encoder
        self.entity_classifier =  entity_classifier
        self.type_classifier = type_classifier
        self.word_pooling = word_pooling

    def forward(self, batch : dict) :
        """
        inference method
        :param batch: batch of data
        :return: batch of data updated with intermediary states
        """
        # context encoding
        batch = self.context_encoder(batch)

        # pooling to word token
        batch = self.word_pooling(batch)

        # entity classification
        batch = self.entity_classifier(batch)
        batch['entity_prediction'] = batch['prediction'].permute(0, 2 ,1)
        batch['entity_prediction_label'] = batch['prediction'].softmax(dim=1).argmax(dim=1)

        # type classification
        batch = self.type_classifier(batch)
        batch['type_prediction'] = batch['prediction'].permute(0, 2, 1)
        batch['type_prediction_label'] = batch['prediction'].softmax(dim=1).argmax(dim=1)

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
    def from_args(cls, args) -> Type[DNERBase]:
        raise NotImplementedError

    def create_samples(self, raw_samples : Any, **kwargs) :
        raise NotImplementedError