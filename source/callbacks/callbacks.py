from __future__ import annotations
from typing import List
from argparse import Namespace, ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from source.utils.register import register
from source.utils.misc import create_or_append_metrics
from pytorch_lightning.callbacks.base import Callback
import pathlib
import json
import torch
import numpy as np
import pickle
import logging

class BaseCallback(Callback) :
    """
    Base class for callbacks.
    """

    @staticmethod
    def add_callback_specific_args(parent_parser):
        return parent_parser

    @classmethod
    def build_from_args(cls, args : Namespace) :
        raise NotImplementedError("Base class callbacks should be inherited")

@register("CALLBACKS")
class EarlyStoppingWrapper(BaseCallback, EarlyStopping) :
    """
    Early stopping wrapper for class from pytorch lightning. Used to be
    with the call back factory.
    """
    _names = ["EarlyStopping"]

    def __init__(self,
                 earlyStopping_monitor : str,
                 patience : int = 10,
                 mode : str = "min") :
        super().__init__(monitor=earlyStopping_monitor,
                         patience=patience,
                         mode=mode)

    @staticmethod
    def add_callback_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        """
        Add early stopping specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('EarlyStopping')

        group.add_argument("--earlyStopping_monitor", type=str,
                           choices=['trainset', 'valset'],
                           default = "valset")
        group.add_argument("--earlyStopping_mode", type=str,
                           choices=['min', 'max'],
                           default="min")
        group.add_argument("--earlyStopping_patience", type=int,
                           default=5)
        return parser

    @classmethod
    def build_from_args(cls, args : Namespace) -> EarlyStoppingWrapper:
        """
        Build Early stopping object from args object issued by the parser
        :param args:
        :return:
        """
        monitor = f"{args.loss}/{args.earlyStopping_monitor}"
        return cls(monitor,
                   args.earlyStopping_patience,
                   args.earlyStopping_mode)

@register("CALLBACKS")
class ModelCheckpointWrapper(BaseCallback, ModelCheckpoint) :
    """
    Model checkpoint wrapper for class from pytorch lightning. Used to be
    with the call back factory.
    """
    _names = ["ModelCheckpoint"]

    def __init__(self,
                 modelCheckpoint_monitor : str,
                 mode : str = "min") :
        super().__init__(monitor=modelCheckpoint_monitor,
                         mode=mode)

    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('modelCheckpoint')

        group.add_argument("--modelCheckpoint_monitor", type=str,
                           choices=['trainset', 'valset'],
                           default = "valset")
        group.add_argument("--modelCheckpoint_mode", type=str,
                           choices=['min', 'max'],
                           default="min")
        return parser

    @classmethod
    def build_from_args(cls, args : Namespace) -> ModelCheckpointWrapper :
        """
        Build Early stopping object from args object issued by the parser
        :param args:
        :return:
        """
        monitor = f"{args.loss}/{args.modelCheckpoint_monitor}"
        return cls(monitor,
                   args.earlyStopping_mode)

@register("CALLBACKS")
class PositionEmbeddingCollector(Callback):
    """
    Callbask for weight scheduling
    """

    def __init__(self, log_dir : pathlib):
        super().__init__()

        self.log_dir = log_dir
        self.data : dict = {}


    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None :

        model = pl_module.model

        self.data['anchor_embeddings'] = model.embeddings.anchor_embeddings

        if hasattr(model.embeddings, "anchor_position") :
            self.data['anchor_position'] = model.embeddings.anchor_position

        if hasattr(model.embeddings, "anchor_position") :
            self.data['pos_embeddings'] = model.embeddings.anchor_position

        if hasattr(model.embeddings, 'initial_anchor_position') :
            self.data['initial_pos_embeddings'] = model.embeddings.initial_anchor_position


        path = self.log_dir / "position_embedding.pickle"
        logging.info(f"Saving position embeddings as  : {path}")

        with path.open("wb+") as file :
            pickle.dump(self.data, file)

    @classmethod
    def build_from_args(cls, args: Namespace) -> PositionEmbeddingCollector:
        """
        Build RotoWireTask1DataCollector object from args object issued by the parser
        :param args: main parser namespace
        :return: RotoWireTask1DataCollector object
        """
        return cls(args.collector_log_dir)


    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('SpanClassificationDataCollector')

        return parser


@register("CALLBACKS")
class GradLearn(Callback):
    """
    Callbask for weight scheduling
    """

    def __init__(self, log_dir : pathlib, layer_names : List[str], unfreeze_limit_epoch : 5):
        super().__init__()

        self.log_dir = log_dir
        self.epoch_accumulator = 0
        self.layer_names = layer_names

        assert unfreeze_limit_epoch > 0
        self.unfreeze_limit_epoch = unfreeze_limit_epoch

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model = pl_module.model

        # freezing model
        model.freeze_model()

        # unfreeze layer of intrests
        model.unfreeze_layers(self.layer_names)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if self.epoch_accumulator == self.unfreeze_limit_epoch :
            pl_module.model.unfreeze_model()

        self.epoch_accumulator += 1

    @classmethod
    def build_from_args(cls, args: Namespace) -> GradLearn:
        """
        Build RotoWireTask1DataCollector object from args object issued by the parser
        :param args: main parser namespace
        :return: RotoWireTask1DataCollector object
        """
        return cls(args.collector_log_dir, args.grad_learn_names, args.grad_learn_unfreeze_epoch)


    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('GradLearn')
        group.add_argument("--grad_learn_names", type=str, nargs="+")
        group.add_argument("--grad_learn_unfreeze_epoch", type=int)
        return parser

@register("CALLBACKS")
class AlternateLearning(Callback):
    """
    Callbask for weight scheduling
    """

    def __init__(self, log_dir : pathlib, layer_names : List[str], alternate_modulo : 5):
        super().__init__()

        self.log_dir = log_dir
        self.epoch_accumulator = 0
        self.layer_names = layer_names
        self.selector = 0
        self.first_epoch = True

        assert alternate_modulo > 1
        self.alternate_modulo = alternate_modulo

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model = pl_module.model

        # unfreeze layer of intrests
        model.freeze_layers(self.layer_names)

        pl_module.model.unfreeze_layers([self.layer_names[self.selector]])
        print(f"UNFREEZING : {[self.layer_names[self.selector]]}")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if not self.first_epoch and self.epoch_accumulator % self.alternate_modulo == 0 :

            # freezing current layer
            pl_module.model.freeze_layers([self.layer_names[self.selector]])
            # update layer to train
            self.selector = index if (index := self.selector + 1) < len(self.layer_names) else 0
            pl_module.model.unfreeze_layers([self.layer_names[self.selector]])

            self.first_epoch = False

        self.epoch_accumulator += 1

    @classmethod
    def build_from_args(cls, args: Namespace) -> AlternateLearning:
        """
        Build RotoWireTask1DataCollector object from args object issued by the parser
        :param args: main parser namespace
        :return: RotoWireTask1DataCollector object
        """
        return cls(args.collector_log_dir, args.alternate_learn_names, args.alternate_modulo)


    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('GradLearn')
        group.add_argument("--alternate_learn_names", type=str, nargs="+")
        group.add_argument("--alternate_modulo", type=int, default=1)
        return parser


@register("CALLBACKS")
class SpanClassificationDataCollector(Callback):
    """
    Callbask for weight scheduling
    """

    def __init__(self, log_dir : pathlib):
        super().__init__()

        self.log_dir = log_dir
        self.data : dict = {}

    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('SpanClassificationDataCollector')

        return parser

    @classmethod
    def build_from_args(cls, args: Namespace) -> SpanClassificationDataCollector:
        """
        Build RotoWireTask1DataCollector object from args object issued by the parser
        :param args: main parser namespace
        :return: RotoWireTask1DataCollector object
        """
        return cls(args.collector_log_dir)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) :

        if dataloader_idx not in self.data :
            self.data[dataloader_idx] = []

        #batch['prediction'] = batch['prediction'].softmax(dim=1).permute(0, 2, 1).tolist()

        """
        print("===============")
        print(f"batch size : {len(batch['spans'])}")
        for k in ['labels_mask', 'prediction', 'labels', 'prediction_label', 'entities'] :
            print(f"{k} : {batch[k][0]}")
        """



        for batch_index, span in enumerate(batch['spans']) :

            obj = { "text" : batch['text'][batch_index] }
            obj["entities"] = batch['entities'][batch_index]

            prediction_shape = batch['prediction'][batch_index].size()
            predictions = batch['prediction'][batch_index]

            if prediction_shape[0] < prediction_shape[1] :
                predictions = torch.transpose(predictions, 0, 1)

            """
            print("====")
            print(f"num label : {predictions}")
            print(f"mask size : {batch['prediction_label'][batch_index].size()}")
            print(f"mask size : {batch['labels_mask'][batch_index].size()}")
            """

            mask_label = batch['labels_mask'][batch_index].bool().unsqueeze(-1)
            predictions = predictions.masked_select(mask_label).tolist()
            labels = batch['labels'][batch_index].masked_select(mask_label).tolist()
            prediction_labels = batch['prediction_label'][batch_index].masked_select(mask_label).tolist()

            """
            print(f"num prediction_labels : {batch['prediction_label'][batch_index].masked_select(mask_label).size()}")
            print(f"num span : {len(span)}")

            print(prediction_labels)
            print(f"num entity : {len(span)}")
            for entity in batch['entities'][0] :
                print(entity)

            exit()
            """

            iterator = enumerate(zip(
                span,
                predictions,
                prediction_labels,
                labels
            ))

            for index_entity, (token_span, proba, prediction_label, label) in iterator  :

                obj["entities"][index_entity]["prediction"] = proba
                obj["entities"][index_entity]['token_span'] = token_span
                obj['entities'][index_entity]['pred_label'] = prediction_label
                obj['entities'][index_entity]['label_intra'] = label

            #print(obj)

            self.data[dataloader_idx].append(obj)

    def on_test_end(self, trainer, pl_module) :

        self.process_test_loss(trainer)

        for idx, data in self.data.items() :

            path = self.log_dir / f"test_{idx}"

            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

            print(f"+ saving results test_{idx} at : {path} ({len(data)} samples)")

            with (path / "output.json").open("w+") as file:
                json.dump(data, file, indent=4)

    def process_test_loss(self, trainer):
        """
        Store the test set losses and update mean and std loss across runs
        :return:
        """

        metrics = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in trainer.logged_metrics.items()
        }

        # if file does not exists create it

        path = self.log_dir / "test_results.json"
        obj = {"mean" : -1, "std" : -1}
        output = {"local" : [metrics]}
        output["global"] = {}
        for loss in metrics.keys() :
            output["global"][loss] = obj

        if not path.exists():
            with path.open("w") as file:
                json.dump(output, file, indent=4)
        else:
            with path.open("r") as file:
                data = json.load(file)

            data["local"].append(metrics)

            stats = np.array([list(metric.values()) for metric in data["local"]])
            means = stats.mean(axis=0)
            stds = stats.std(axis=0)

            for mean, std, loss in zip(means, stds, data['global'].keys()) :
                data["global"][loss]["mean"] = mean
                data["global"][loss]["std"] = std

            with path.open("w") as file:
                json.dump(data, file, indent=4)

@register("CALLBACKS")
class SequenceTaggingDataCollector(Callback):
    """
    Callbask for weight scheduling
    """

    def __init__(self, log_dir : pathlib, store_results : bool = False):
        super().__init__()

        self.log_dir = log_dir
        self.data : List[List] = [[], [], [], [], [], [], []]

        self.store_results = store_results

    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('modelCheckpoint')

        group.add_argument(
            "--collector_log_dir",
            type=pathlib.Path,
            default=pathlib.Path(__file__).parents[2] / "data" / "inference" )

        group.add_argument("--store_results", action='store_true', default=False)
        return parser

    @classmethod
    def build_from_args(cls, args: Namespace) -> SequenceTaggingDataCollector :
        """
        Build RotoWireTask1DataCollector object from args object issued by the parser
        :param args: main parser namespace
        :return: RotoWireTask1DataCollector object
        """
        return cls(args.collector_log_dir)


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) :

        if self.store_results :
            batch['prediction'] = batch['prediction'].softmax(dim=1).permute(0, 2, 1).tolist()
            for index in range(len(batch['text'])) :

                obj = { "text" : batch['text'][index] }
                obj["entities"] = batch['entities'][index]
                obj['prediction_label'] = batch['prediction_label'][index].tolist()
                obj['prediction'] = batch['prediction'][index]
                obj['labels'] = batch['labels'][index].tolist()
                obj['spans'] = batch['spans'][index]

                self.data[dataloader_idx].append(obj)
        else :
            print("NO STORE")


    def on_test_end(self, trainer, pl_module) :

        self.process_test_loss(trainer)

        for idx, data in enumerate(self.data):

            if data == [] :
                continue

            path = self.log_dir / f"test_{idx}"

            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

            print(f"+ saving results test_{idx} at : {path}")

            with (path / "output.json").open("w+") as file:
                json.dump(data, file, indent=4)

    def process_test_loss(self, trainer):
        """
        Store the test set losses and update mean and std loss across runs
        :return:
        """

        metrics = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in trainer.logged_metrics.items()
        }

        # if file does not exists create it

        path = self.log_dir / "test_results.json"
        obj = {"mean" : -1, "std" : -1}
        output = {"local" : [metrics]}
        output["global"] = {}
        for loss in metrics.keys() :
            output["global"][loss] = obj

        if not path.exists():
            with path.open("w") as file:
                json.dump(output, file, indent=4)
        else:
            with path.open("r") as file:
                data = json.load(file)

            data["local"].append(metrics)

            stats = np.array([list(metric.values()) for metric in data["local"]])
            means = stats.mean(axis=0)
            stds = stats.std(axis=0)

            for mean, std, loss in zip(means, stds, data['global'].keys()) :
                data["global"][loss]["mean"] = mean
                data["global"][loss]["std"] = std

            with path.open("w") as file:
                json.dump(data, file, indent=4)

@register("CALLBACKS")
class BARTNERDataCollector(SequenceTaggingDataCollector):
    """
    Callbask for weight scheduling
    """

    def __init__(self, log_dir : pathlib):
        super().__init__(log_dir)

        self.log_dir = log_dir
        self.data : List[List] = [[], [], [], [], [], [], []]

    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('modelCheckpoint')

        group.add_argument(
            "--collector_log_dir",
            type=pathlib.Path,
            default=pathlib.Path(__file__).parents[2] / "data" / "inference" )

        return parser

    @classmethod
    def build_from_args(cls, args: Namespace) -> SequenceTaggingDataCollector :
        """
        Build RotoWireTask1DataCollector object from args object issued by the parser
        :param args: main parser namespace
        :return: RotoWireTask1DataCollector object
        """
        return cls(args.collector_log_dir)


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) :

        batch['prediction'] = batch['prediction'].softmax(dim=1).permute(0, 2, 1).tolist()
        for index in range(len(batch['text'])) :

            obj = { "text" : batch['text'][index] }
            obj["entities"] = batch['entities'][index]
            obj['prediction_label'] = batch['prediction_label'][index].tolist()
            obj['prediction'] = batch['prediction'][index]
            obj['labels'] = batch['labels'][index].tolist()

            self.data[dataloader_idx].append(obj)


@register("CALLBACKS")
class ValidationDataCollector(Callback):
    """
    Callbask for weight scheduling
    """

    def __init__(self, log_dir : pathlib):
        super().__init__()

        self.log_dir = log_dir
        self.data : List = [[]]
        print(f"## INFO : Validation data stored at {log_dir}")

    @staticmethod
    def add_callback_specific_args(parent_parser):
        """
        Add checkpoint specifics args to the main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('ValidationDataCollector')

        group.add_argument(
            "--validation_collector_log_dir",
            type=pathlib.Path,
            default=pathlib.Path(__file__).parents[2] / "data" / "inference" )

        return parser

    @classmethod
    def build_from_args(cls, args: Namespace) -> ValidationDataCollector :
        """
        Build RotoWireTask1DataCollector object from args object issued by the parser
        :param args: main parser namespace
        :return: RotoWireTask1DataCollector object
        """
        return cls(args.collector_log_dir)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) :


        for index in range(len(batch['texts'])) :

            obj = {"text" : batch['texts'][index] }
            obj['loss_sample'] = outputs['loss_sample'].tolist()
            obj["mentioned_players"] = batch['mentioned_players_list'][index]
            obj['prediction_label'] = outputs['output']['prediction_label'][index].tolist()
            obj['labels'] = outputs['output']['labels'][index].tolist()

            self.data[-1].append(obj)


    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None :
        self.data.append([])

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if self.data == [[]] :
            return

        path = self.log_dir / f"validation_data"

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        else :
            return

        print(f"+ saving validation data at : {path}")

        with (path / "validation_data.json").open("w+") as file:
            json.dump(self.data, file, indent=4)

