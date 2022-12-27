from __future__ import annotations
from argparse import ArgumentParser, Namespace
from source.utils.register import register
from typing import List, Type, Set, Any, Tuple, Optional
from source.data.data_modules.DNERBaseDataModule import DNERBaseDataModule
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import logging

@register("DATASETS")
class ImDBDataModule(DNERBaseDataModule):
    """
    Data Module for handling RotoWire Task 1 Data.
    """
    def __init__(self,
                 max_sequence_length : int = 512,
                 limit_credit: Optional[Tuple[int, int]] = None,
                 **kwargs) :
        """
        """

        super().__init__(**kwargs)

        self.task = kwargs['task']

        self.limit_credit = limit_credit

        self._create_sample_param['padding'] = max_sequence_length

    @classmethod
    def from_args(cls, args : Namespace) -> ImDBDataModule :
        """
        Build Rotowire Task 1 Datamodule from parser Namspace args
        :param args: namespace from main parser
        :return: RotoWireTask1DataModule object
        """
        return cls(**vars(args))

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add Data Module specific args to the main parser
        :param parent_parser: main parser
        :return: updated main parser
        """
        parent_parser = DNERBaseDataModule.add_data_specific_args(parent_parser)

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('CNET-ImDB DataModule')
        group.add_argument("--limit_credit", type=int, default=None, nargs="+")
        group.add_argument("--max_sequence_length", type=int, default=512)


        return parser

    def filter_data_by_credit(self, data : List[dict], name : str = "") -> List[dict] :

        def get_number_credit(actor_list) -> int :
            return max([actor['label'] for actor in actor_list])

        if data is None :
            raise ValueError(f"Empty dataset")

        data = [
            sample for sample in data
            if self.limit_credit[0] <= get_number_credit(sample['entities']) <= self.limit_credit[1]
        ]

        logging.info(f"-> {len(data)} {name} samples remaining")

        return data

    def preprocess_sets(self) -> None :

        # filtering credit
        logging.info(f"Filtering out sample having label outside {self.limit_credit} credits")
        if self.limit_credit is not None :
            self.train_samples = self.filter_data_by_credit(self.train_samples, name = "train")

            self.val_samples = self.filter_data_by_credit(self.val_samples, name = "train")

            self.test_data = [self.filter_data_by_credit(data, name = "test") for data in self.test_data]


    def process_weights(self, data : List[dict]) -> None :
        if self.task == 'SequenceTaggingTask' :
            self.process_weights_CNER(data)
        elif self.task == 'SpanClassificationTask' :
            self.process_weights_CNET(data)
        else :
            RuntimeError(f"Not supported task {self.task}")

    # override process weights
    def process_weights_CNET(self, data : List[dict]) -> None :
        """
        process weight for each class
        :param data: list of dict representing games
        :return: list of float, one for each class
        """

        logging.info("Processing weights...")

        list_label = []

        for sample in tqdm(data) :
            for player in sample['entities'] :
                list_label.append(player['label'])

        unique_label = np.unique(list_label)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_label,
            y=list_label
        )

        # storing weight information
        for label in unique_label :
            logging.info(f"label {label} : {weights[label]} ({len([x for x in list_label if x == label])} samples)")

        self._create_sample_param['weights'] = weights



    def process_weights_CNER(self, samples : List[Any]) -> None :

        def get_label_actor(actor: dict) -> List[int]:
            """
            Convert a token sequence representing a player by its corresponding
            label sequence
            :param player: name of the player (string)
            :return: list of label
            """

            len_tokens = len(actor['name'].split(" "))

            return [actor['label'] + 1] * (len_tokens)

        labels_list = []

        for sample in tqdm(samples) :

            len_text = len(sample['text'].split(" "))

            for player in sample['entities'] :
                labels_player = get_label_actor(player)
                len_text -= len(labels_player)
                labels_list += labels_player

            labels_list += [0] * len_text

        label_unique = np.unique(labels_list)
        weights = compute_class_weight(class_weight="balanced", classes=label_unique, y=labels_list)

        w_list_str = '\n'.join([f'\to {c} : {w}' for c, w in zip(label_unique, weights)])
        logging.info(f"+ inferred weights : {w_list_str}")

        self._create_sample_param['weights'] = weights

