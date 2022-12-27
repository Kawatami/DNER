from __future__ import annotations

import warnings
from argparse import ArgumentParser, Namespace
from typing import Type, Optional, Callable
from source.data.data_modules.base_datamodule import BaseDataModule
from source.utils.register import register
from source.data.datasets.DNERDataset import DNERDataset
from source.data.utils.training_batch import collect
from torch.utils.data.dataloader import DataLoader
import json, sys
from typing import List, Type, Set, Any, Tuple, Optional
import pathlib
import numpy as np
from source.data.preprocessing.preprocessing import resolve_text_preprocessing, text_preprocessing
from source.data.tokenizer.tokenizer import resolve_tokenizer, tokenizers
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import logging
import pickle
#import hickle
from sys import getsizeof

@register("DATASETS")
class DNERBaseDataModule(BaseDataModule):
    """
    Data Module for handling RotoWire Task 1 Data.
    """
    def __init__(self,
                 collector_log_dir: pathlib.Path,
                 train_files : List[str],
                 test_files : List[str],
                 text_preprocessing : List[Callable] = [],
                 label_preprocessing : List[Callable] = [],
                 batch_size : int = 32,
                 verbose : int = logging.INFO,
                 seed : int = 42,
                 data_tag : Optional[str]  = None,
                 storage_mode : str = "pickle",
                 process_weights : bool = True,
                 force_json_loading : bool = False,
                 **kwargs) :
        """
        """

        # base class init
        super().__init__(**kwargs)

        # create log dir
        if not collector_log_dir.exists() :
            collector_log_dir.mkdir(parents=True, exist_ok=True)

        # Init logger
        logging.basicConfig(
            filename=collector_log_dir / "logs.log",
            encoding='utf-8',
            level=verbose
        )
        # adding standard output level logging
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info("DataModule init...")


        self.storage_mode = storage_mode
        # pickles info
        self.data_tag = data_tag
        self.pickle_load = False
        self.force_json_loading = force_json_loading
        if force_json_loading :
            logging.warn(f"WARNING : force json loading enabled. Pickle will not be loaded.")

        # batch size info
        assert 0 < batch_size
        self.batch_size = batch_size

        # set info
        self.sets : dict = {
            "train" : None,
            "val" : None,
            "test" : []
        }

        # storing preprocessing info
        self.text_preprocessing = text_preprocessing
        self.label_preprocessing = label_preprocessing

        self.seed = seed
        self.weights = process_weights

        # creating data collection directory id needed
        self.train_files = train_files
        self.test_files = test_files


        self.collector_log_dir = collector_log_dir

        logging.info("DONE")


    @classmethod
    def from_args(cls, args : Namespace) -> DNERBaseDataModule :
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

        parent_parser = BaseDataModule.add_data_specific_args(parent_parser)


        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('CNET base module')

        group.add_argument("--batch_size", type = int, default=32)
        group.add_argument("--text_preprocessing", type=resolve_text_preprocessing, nargs="+",
                           default=[], choices=text_preprocessing.values())
        group.add_argument("--verbose", type = int, default=logging.INFO)
        group.add_argument("--process_weights", action="store_true",
                           default=True)
        group.add_argument("--force_json_loading", action = "store_true", default=False)
        group.add_argument("--data_tag", type=str, default=None)
        group.add_argument("--storage_mode", type=str, default="pickle",
                           choices = ["pickle", "hickle", "chunked_hickle"])

        group.add_argument("--train_files", type=list, nargs="+",
                           default=['train.json', 'val.json'])
        group.add_argument("--test_files", type=str, nargs="+",
                           default=['seen_unseen.json', 'unseen.json', 'seen.json', 'all.json'])

        return parser

    def _get_dataloader(self, split='train') -> Type[DataLoader]:
        """
        Getter for sets DataLoader
        :param split: split to choose from ("train", "val", "test")
        :return:
        """
        assert split in ["train", "val", "test"]
        return self.sets[split]

    def preprocess_sets(self) -> None:
        """
        Preprocessing to be applied on set level
        """
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split raw_data into three separate sets. This is done in a way that teams
        in the test set don't overlap with teams in the train set
        """

        if self.pickle_load :
            logging.info(f"Pickle loading, skipping set up phase")
            return # if data are loaded via pickle then do nothing

        # processing weights
        if self.weights:
            weights = self.process_weights(self.train_samples)
        else:
            weights = [1.0, 1.0]

        # set level preprocessing
        self.preprocess_sets()

        logging.info("+ Creating DataLoader...")
        self.sets["train"] = self.create_dataLoader(
            self.train_samples,
            weights=weights,
            shuffle=True,
            text_preprocessing=self.text_preprocessing,
            label_preprocessing=self.label_preprocessing
        )
        if self.storage_mode == "chunked_hickle" :
            self.save_chuncked_hickles(self.sets['train'], "train")

        self.sets["val"] = self.create_dataLoader(self.val_samples)
        if self.storage_mode == "chunked_hickle" :
            self.save_chuncked_hickles(self.sets['val'], "val")

        for data in self.test_data :
            self.sets["test"].append(self.create_dataLoader(data))
        if self.storage_mode == "chunked_hickle" :
            self.save_chuncked_hickles(self.sets['test'], "test")


        if self.data_tag is not None :
            if self.storage_mode == "pickle" :
                self.save_sets_pickles()
            elif self.storage_mode == "hickle" :
                self.save_sets_hickles()
            else :
                raise ValueError(f"{self.storage_mode} unsupported type of storage mode")
        else :
            logging.warn("No data tag provided to store datasets in pickle file.")

        logging.info("DONE")


    def save_sets_pickles(self) -> None :
        if self.sets['train'] is None :
            logging.critical(f"Saving pickles ERROR : sets not initialized")
            raise ValueError(f"Sets seems to be not initaliazed ?")
        else :
            path = self.data_directory / f"{self.data_tag}.pickle"
            with path.open("wb") as file :
                logging.info(f"Storing pickles dataset at {path} {getsizeof(self.sets)} bytes")
                pickle.dump(self.sets, file)

    def save_sets_hickles(self) -> None :
        if self.sets['train'] is None :
            logging.critical(f"Saving pickles ERROR : sets not initialized")
            raise ValueError(f"Sets seems to be not initaliazed ?")
        else :
            path = self.data_directory / f"{self.data_tag}.hickle"
            logging.info(f"Storing hickles dataset at {path} {getsizeof(self.sets)} bytes")
            #hickle.dump(self.sets, path, mode="w")

    def save_chuncked_hickles(self, set, name) -> None :
        if self.sets['train'] is None :
            logging.critical(f"Saving pickles ERROR : sets not initialized")
            raise ValueError(f"Sets seems to be not initaliazed ?")
        else :

            path = self.data_directory / f"{self.data_tag}"

            path.mkdir(parents=True, exist_ok=True)

            path /= f"{name}.hickle"
            logging.info(f"Storing hickles dataset at {path} {getsizeof(set)} bytes")
            hickle.dump(set, path, mode="w")


    def prepare_data(self, *args, **kwargs) -> None :
        """
        Load data from the [data_directory] attribute. If a [data_tag] is provided, data loading via pickle file is
        attempted. In case of failure standard loading via json file is tried out.
        """


        def check_and_load(path : pathlib.Path, mode : str = "r") -> Any :
            """
            Check the path and load the data
            """

            if not path.exists():
                msg = f"{path} not found."
                logging.warn(msg)
                warnings.warn(msg)
                return None

            with path.open(mode) as file:
                data = json.load(file)

            print(f"{path.stem} samples : {len(data)}")

            return data

        logging.info(f"Loading data at {self.data_directory}")

        if not self.data_directory.exists() :
            msg = f"{self.data_directory} not found"
            logging.critical(msg)
            raise ValueError(msg)

        elif not self.force_json_loading :
            if self.data_tag is not None :
                if self.storage_mode == "pickle" :
                    path = self.data_directory / f"{self.data_tag}.pickle"
                    logging.info(f"Data tag provided trying to load pickle at {path}")
                    if path.exists() :
                        logging.info(f"Loading via pickle file at : {path}")
                        with path.open("rb") as file :
                            self.sets = pickle.load(file)

                        self.pickle_load = True
                        return
                    else :
                        logging.error(f"{path} not found.")
                elif self.storage_mode == "hickle" :
                    path = self.data_directory / f"{self.data_tag}.hickle"
                    logging.info(f"Data tag provided trying to load hickle at {path}")
                    if path.exists():
                        logging.info(f"Loading via hickle file at : {path}")
                        self.sets = hickle.load(path)
                        self.pickle_load = True
                        return
                    else:
                        logging.error(f"{path} not found.")
                elif self.storage_mode == "chuncked_hickle" :
                    path = self.data_directory / f"{self.data_tag}"
                    logging.info(f"Data tag provided trying to load chuncked hickle at {path}")
                    if path.exists():
                        for set in self.sets.keys() :
                            path_file = path / f"{set}.hickle"
                            logging.info(f"Loading via hickle file at : {path_file}")
                            self.sets = hickle.load(path_file)
                        self.pickle_load = True
                        return
                    else:
                        logging.error(f"{path} not found.")
        logging.info(f"Loading via json file at : {self.data_directory}")

        # train files
        self.train_samples = check_and_load(self.data_directory / "train" / self.train_files[0])

        # validation
        self.val_samples = check_and_load(self.data_directory / "train" / self.train_files[1])

        # test data
        self.test_data = []

        for file in self.test_files :

            path_file = self.data_directory / "test" / file
            data = check_and_load(path_file)

            if len(data) <= 0 :
                msg = f"Warning : data loading from {path_file} is empty"
                logging.warn(msg)

            if data is not None :
                self.test_data.append(data)

        if self.test_data == [] :
            raise RuntimeError(f"ERROR : No test data could be loaded.")

    def process_weights(self, data : List[dict]) -> List[float] :
        """
        process weight for each class
        :param data: list of dict representing games
        :return: list of float, one for each class
        """
        raise NotImplementedError

    def create_dataLoader(self,
                          data : List[Any],
                          weights : Optional[List[float]] = [1.0] * 8,
                          text_preprocessing: List[Callable] = [],
                          label_preprocessing: List[Callable] = [],
                          shuffle : bool = False) -> DataLoader :
        """
        Create a dataloader for the given samples
        :param data: sample to to wrap in the dataLoader
        :param name: name of the dataset
        :param weights: weight used for loss
        :return: Dataloader
        """

        # create dataset
        ds = DNERDataset(
            text_preprocessing=text_preprocessing,
            label_preprocessing=label_preprocessing,
            samples_creator = self.create_sample_method
        )

        #print(f"BEFORE num samples : {len(data)}")


        ds.process_data(data, **self._create_sample_param)

        #print(f"batch size : {self.batch_size}")
        #exit()


        res = DataLoader(
            ds,
            collate_fn=collect,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True
        )

        #print(f"AFTER num samples : {len(res) * self.batch_size}")

        # create dataloader
        return res



