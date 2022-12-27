import torch
from source.metrics.base_metric import BaseMetric
from source.utils.register import register
from typing import Any, List, Tuple, Union
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from enum import Enum

@register('METRICS')
class Accuracy(BaseMetric):
    """
    Process Accuracy
    """
    _names = ['Accuracy', 'Acc']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, acc_threshold : float, **kwargs):
        super().__init__(**kwargs)

        assert acc_threshold > 0.0
        self.threshold = acc_threshold

        self.add_state("prediction", default=torch.empty(0), dist_reduce_fx='cat')
        self.add_state("label", default=torch.empty(0), dist_reduce_fx='cat')

    def update(self, batch : dict) -> None:
        prd, tgt = self._prep_inputs(batch)
        assert prd.size() == tgt.size(), f" prd ({prd.size()}) and tgt ({tgt.size()}) should have the same size"

        mask = batch['labels_mask'].bool().cpu()
        prd = prd.masked_select(mask) > self.threshold
        tgt = tgt.masked_select(mask)


        # updating state
        self.prediction = torch.cat((self.prediction, prd))
        self.label = torch.cat((self.label, tgt))

    def compute(self) -> Any :
        return (self.prediction == self.label).float().mean()

    @staticmethod
    def add_metric_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Accuracy metric')
        group.add_argument("--acc_threshold", type=float, default=0.5)

        return parser

    def _prep_inputs(self, model_outputs):
        prd, tgt = model_outputs['prediction'].squeeze(-1).float().cpu() , model_outputs['labels'].cpu()

        return prd, tgt

@register('METRICS')
class AccuracyMultiClass(BaseMetric):
    _names = ['AccuracyMultiClass', 'AccMC']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, accM_include_background : bool = False, **kwargs):
        super().__init__(**kwargs)
        self.add_state("prediction", torch.empty(0, device=self.device), dist_reduce_fx='cat')
        self.add_state("label", torch.empty(0, device=self.device), dist_reduce_fx='cat')
        self.include_background = accM_include_background

    def update(self, batch : dict) -> None :
        prd, tgt = self._prep_inputs(batch)

        if not self.include_background :
            # processing mask to ignore background label 0
            mask_prd_background = prd != 0
            mask_tgt_background = tgt != 0
            mask_background = torch.logical_or(mask_tgt_background, mask_prd_background)

            mask = torch.logical_and(mask_background, batch['labels_mask'].bool())
        else :
            mask = batch['labels_mask'].bool()

        #print(f"prd : {prd.size()}")
        #print(f"tgt : {tgt.size()}")
        #print(f"mask : {mask.size()}")


        # not selecting backgroud label 0
        prd = prd.masked_select(mask).cpu()
        tgt = tgt.masked_select(mask).cpu()

        # updating state
        self.prediction = torch.cat((self.prediction, prd))
        self.label = torch.cat((self.label, tgt))

    def compute(self) -> Any:
        return (self.prediction  == self.label).float().mean()

    @staticmethod
    def add_metric_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Accuracy metric')
        group.add_argument("--accM_include_background", default=False, action='store_true')

        return parser


    def _prep_inputs(self, batch):
        prd, tgt = batch['prediction_label'], batch['labels']

        return prd, tgt

@register('METRICS')
class CNETClassificationReport(BaseMetric):

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("prediction", [], dist_reduce_fx='cat')
        self.add_state("label", [], dist_reduce_fx='cat')

    def update(self, batch : dict) -> None :
        prd, tgt = self._prep_inputs(batch)

        mask = batch['labels_mask'].bool().view(-1).cpu()

        prd = prd.masked_select(mask)
        tgt = tgt.masked_select(mask)

        # updating state
        self.prediction += prd.tolist()
        self.label += tgt.tolist()

    def compute(self) -> Any:


        classification_data = classification_report(self.label, self.prediction, output_dict=True)

        res = []

        for label, data in classification_data.items() :

            if isinstance(data, dict) :
                for name, metric in data.items() :
                    res.append((f"{name}_{label}", metric))
            else :
                res.append((f"classif_{label}", data))

        return res

    @staticmethod
    def add_metric_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Accuracy metric')

        return parser


    def _prep_inputs(self, batch):
        prd, tgt = batch['prediction_label'].view(-1).cpu(), batch['labels'].view(-1).cpu()



        return prd, tgt

class CNERClassificationReportBase(BaseMetric):

    def __init__(self, classes : List[Any], ignore_label : bool = False, **kwargs):
        super().__init__(**kwargs)

        print(f"classes : {classes}")

        assert classes != [], "CNERClassificationReportBase ERROR : No class list provided for Entity report !"
        self.classes = classes

        self.ignore_label = ignore_label

        self.epsilone = 1e-6

        for label in classes :
            self.add_state(f"TP_{label}", torch.tensor([0]), dist_reduce_fx='sum')
            self.add_state(f"FN_{label}", torch.tensor([0]), dist_reduce_fx='sum')
            self.add_state(f"FP_{label}", torch.tensor([0]), dist_reduce_fx='sum')

    def add_state_name(self, name: str, label : Union[int, str], value : int) -> None :
        state_name = f"{name}_{label}"
        if not hasattr(self, state_name) :
            raise ValueError(f"State {state_name} not registered !")
        else :
            value = getattr(self, state_name) + value
            setattr(self, state_name, value)

    def collect_entity(self, label_sequence : List[int]) -> List[Tuple[int, int, int]]:
        """
        Collect detected entity
        """

        # defining internal states
        class States(Enum) :
            INSIDE_ENTITY = 0,
            SEARCHING = 1

        res = []
        begin = 0

        # Helper functions

        def get_label(label : int) -> int :
            """
            Get the global given a token
            """
            if label == 0 :
                raise RuntimeError(f"get label ERROR : label 0 does not correspond to an entity !")
            else :
                return (label - 1) // 4

        def is_begin(label : int) -> bool :
            return (label - 1) % 4 == 0

        def is_inside(label : int) -> bool :
            return (label - 1) % 4 == 1

        def is_end(label : int) -> bool :
            return (label - 1) % 4 == 2

        def is_single(label : int) -> bool :
            return (label - 1) % 4 == 3

        global curr_label
        global state
        curr_label = -1
        state = States.SEARCHING

        def reset_state() -> None :
            global curr_label
            global state
            curr_label = -1
            state = States.SEARCHING

        for index, label in enumerate(label_sequence):

            if state == States.SEARCHING :
                # if state searching

                if label == 0 :
                    # if not an entity continue
                    continue

                elif is_begin(label) :
                    # if curr is begining then change state and store current label
                    begin = index
                    state = States.INSIDE_ENTITY
                    curr_label = label

                elif is_single(label) :
                    # if curr is single then store entity and reset state
                    res.append((index, index, get_label(label)))


            elif state == States.INSIDE_ENTITY :
                # if state inside an entity

                if is_inside(label):
                    # if curr is inside and correspond to the begin label continue
                    if curr_label != -1 and label == curr_label + 1:
                        continue
                    else:  # else this is an error reset state
                        reset_state()

                elif is_end(label) :
                    # if curr is end and correspond to the begin label store entity
                    if curr_label != -1 and label == curr_label + 2 :
                        res.append((begin, index, get_label(label)))
                    reset_state()
                else :
                    reset_state()

        return res

    def strict_matching(self, e1 : Tuple[int, int, int], e2 : Tuple[int, int, int]) -> bool :
        """
        Match entities by checking every entry
        """
        for i in range(len(e1)):
            if e1[i] != e2[i]:
                return False
        return True


    def process_samples(self, ref : List[Tuple], pred : List[Tuple]) -> dict:
        pred = set(pred)
        ref = set(ref)

        FN = ref - pred
        TP = ref.intersection(pred)
        FP = pred - ref

        return {
            "FN" : FN,
            "TP": TP,
            "FP" : FP
        }

    def process_classification_data(self,
                                    entities : List[Tuple[int, int, int]],
                                    ref : List[Tuple[int, int, int]]) -> None:
        """
        Process false positive...
        """

        # strict matching
        samples_data = self.process_samples(ref, entities)
        for type, set in samples_data.items() :
            for entity in set :
                self.add_state_name(type, entity[-1], 1)



        """
        # detection TP and FP
        for entity in entities:
            strict_match = False
            for ref_entity in ref :
                if self.strict_matching(entity, ref_entity):  # True positive
                    strict_match = True
                    break

            if strict_match :
                self.add_state_name("TP", entity[-1], 1)
            else:
                self.add_state_name("FP", entity[-1], 1)

        # detection FN
        for ref_entity in ref :
            strict_match = False
            for entity in entities :
                if self.strict_matching(entity, ref_entity):  # True positive
                    strict_match = True

            if not strict_match:
                self.add_state_name("FN", ref_entity[-1], 1)
        """

    def update(self, batch : dict) -> None :
        """
        update metric
        """
        label_gt = batch['labels'].tolist()
        prediction_labels = batch['prediction_label'].tolist()

        for label, prediction_label in zip(label_gt, prediction_labels) :
            gt_entities = self.collect_entity(label)
            predicted_entity = self.collect_entity(prediction_label)
            self.process_classification_data(predicted_entity, gt_entities)

    def compute(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Process metric final result
        """
        res = []
        micro_TP = 0
        micro_FP = 0
        micro_FN = 0

        for label in self.classes :

            # retreive data
            TP, FP, FN = getattr(self, f"TP_{label}"), getattr(self, f"FP_{label}"), getattr(self, f"FN_{label}")

            # adding for micro metric processing
            micro_TP += TP
            micro_FP += FP
            micro_FN += FN

            # processing metric per class
            precision = TP / (TP + FP + self.epsilone)
            recall = TP / (TP + FN + self.epsilone)
            F1 = 2 * ((precision * recall) / (precision + recall + self.epsilone))

            # adding to resulting list
            res.append((f"precision_{label}", precision.cuda()))
            res.append((f"recall_{label}", recall.cuda()))
            res.append((f"F1_{label}", F1.cuda()))

        if not self.ignore_label :
            # processing metric per class
            precision = micro_TP / (micro_TP + micro_FP + self.epsilone)
            recall = micro_TP / (micro_TP + micro_FN + self.epsilone)
            F1 = 2 * ((precision * recall) / (precision + recall + self.epsilone))

            # adding to resulting list
            res.append((f"precision_micro", precision))
            res.append((f"recall_micro", recall))
            res.append((f"F1_micro", F1))

        return res

@register("METRICS")
class CNERClassificationReport(CNERClassificationReportBase) :
    def __init__(self, classes : List[int], **kwargs) :
        super().__init__(classes, **kwargs)

    @staticmethod
    def add_metric_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('CNERClassificationReport metric')
        group.add_argument("--classes", type=int, default=[], nargs="+")

        return parser

@register("METRICS")
class BARTNERClassificationReport(CNERClassificationReportBase) :
    def __init__(self, classes : List[int], **kwargs) :
        super().__init__(classes, **kwargs)

    @staticmethod
    def add_metric_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('CNERClassificationReport metric')
        group.add_argument("--classes", type=int, default=[], nargs="+")

        return parser

    def collect_entity_BARTNER(
            self,
            label_sequence: List[int],
            sequence_lenght,
            attention_mask
    ) -> List[Tuple[int, int, int]]:
        """
        Collect detected entity
        """

        res = []
        accumulator = []

        #print(sequence_lenght)

        for label, att_mask in zip(label_sequence, attention_mask) :

            if att_mask == 0 :
                break

            if label >= sequence_lenght - 1 :
                if (class_index := label - sequence_lenght) < len(self.classes) and len(accumulator) >= 2: # if not it means the predicted label is outside the
                    current_label = self.classes[class_index] # the input sequence
                    res.append((accumulator[0], accumulator[-1], current_label))
                accumulator = []
            else :
                accumulator.append(label)

        #print(res)

        return res

    def update(self, batch : dict) -> None :
        """
        update metric
        """
        labels, prediction_labels = batch['labels'], batch['prediction_label']
        sequence_length = batch['attention_mask'].size(1)
        attention_masks = batch['attention_mask']

        for label, prediction_label, att_mask in zip(labels, prediction_labels, attention_masks) :
            predicted_entity = self.collect_entity_BARTNER(
                prediction_label.tolist(),
                sequence_length,
                att_mask
            )
            gt_entities = self.collect_entity_BARTNER(
                label.tolist(),
                sequence_length,
                att_mask
            )
            self.process_classification_data(predicted_entity, gt_entities)


@register("METRICS")
class BARTNERClassificationReportEntity(BARTNERClassificationReport) :
    def __init__(self, **kwargs) :
        super().__init__(['entity'], ignore_label = True)

    def process_classification_data(self,
                                    entities : List[Tuple[int, int, int]],
                                    ref : List[Tuple[int, int, int]]) -> None:
        """
        Process false positive...
        """

        # strict matching
        samples_data = self.process_samples(ref, entities)
        for type, set in samples_data.items():
            self.add_state_name(type, "entity", len(set))

    def collect_entity_BARTNER(
            self,
            label_sequence: List[int],
            sequence_lenght,
            attention_mask
    ) -> List[Tuple[int, int]]:
        """
        Collect detected entity
        """

        res = []
        accumulator = []

        print(sequence_lenght)

        for label, att_mask in zip(label_sequence, attention_mask) :

            if att_mask == 0 :
                break

            if label >= sequence_lenght :
                res.append((accumulator[0], accumulator[-1],))
                accumulator = []
            else :
                accumulator.append(label)

        return res

    def strict_matching(self, e1 : Tuple[int, int, int], e2 : Tuple[int, int, int]) -> bool :
        """
        Match entities by checking every entry. Skipping the label class
        """
        for i in range(len(e1) - 1):
            if e1[i] != e2[i]:
                return False
        return True

@register("METRICS")
class CNERClassificationReportEntity(CNERClassificationReportBase) :
    def __init__(self, **kwargs) :
        super().__init__(['entity'], ignore_label = True)

    @staticmethod
    def add_metric_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('CNERClassificationReport Entity')

        return parser

    def process_samples(self, ref : List[Tuple], pred : List[Tuple]) -> dict:

        # removing label from entities
        pred = [(x[0], x[1]) for x in pred]
        ref = [(x[0], x[1]) for x in ref]

        return super().process_samples(ref, pred)

    def process_classification_data(self,
                                    entities : List[Tuple[int, int, int]],
                                    ref : List[Tuple[int, int, int]]) -> None:
        """
        Process false positive...
        """
        # strict matching
        samples_data = self.process_samples(ref, entities)
        for type, set in samples_data.items() :
            self.add_state_name(type, "entity", len(set))

        """
        # detection TP and FP
        for entity in entities:
            strict_match = False
            for ref_entity in ref :
                if self.strict_matching(entity, ref_entity):  # True positive
                    strict_match = True
                    break

            if strict_match :
                self.add_state_name("TP", "entity", 1)
            else:
                self.add_state_name("FP", "entity", 1)

        # detection FN
        for ref_entity in ref :
            strict_match = False
            for entity in entities :
                if self.strict_matching(entity, ref_entity):  # True positive
                    strict_match = True

            if not strict_match:
                self.add_state_name("FN", "entity", 1)
        """

    def strict_matching(self, e1 : Tuple[int, int, int], e2 : Tuple[int, int, int]) -> bool :
        """
        Match entities by checking every entry. Skipping the label class
        """
        for i in range(len(e1) - 1):
            if e1[i] != e2[i]:
                return False
        return True