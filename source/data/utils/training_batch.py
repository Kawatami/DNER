from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import torch
from source.data.preprocessing.onfly_preprocessing import random_masking
from source.data.preprocessing.OnFlyPreprocessor import OnFlyPreprocessor
from source.data.data_sample.DataSample import DataSample, BatchDict
spanType = Tuple[int, int]
spanList = List[Tuple[int, int]]


@dataclass
class SpanClassificationTrainingSample :
    """
    Class representating a RotoWire Task 1 sample data
    """
    input_ids : List[int]
    attention_mask : List[int]
    spanList : spanList
    labels : List[int]
    weights : List[float]
    text : str
    mentions : List[dict]

@dataclass
class SequenceTaggingTrainingSample :
    """
    Class representation a RotoWire Task 2 sample data in the sequential case
    """
    input_ids : List[int]
    attention_mask : List[int]
    labels_sequence : List[int]
    weights_sequence : List[float]
    label_mask: List[int]
    text : str
    span_list : List[Tuple[int, int]]
    mentions : List[dict]

@dataclass
class RotoWireTask2TrainingSampleJoint :
    """
    Class representation a RotoWire Task 2 sample data in the sequential case
    """
    input_ids : List[int]
    attention_mask : List[int]
    labels_sequence : List[int]
    weights_sequence : List[float]
    text : str
    span_list : List[Tuple[int, int]]
    mentioned_players : List[dict]

@dataclass
class ImDBTask2TrainingSampleJoint :
    """
    Class representation a Imdb Task 2 sample data in the sequential case
    """
    input_ids : List[int]
    attention_mask : List[int]
    labels_sequence : List[int]
    weights_sequence : List[float]
    text : str
    span_list : List[Tuple[int, int]]
    mentioned_players : List[dict]

def build_string_from_spans(sample):

    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    spans = sample.span_list
    masks = sample.label_mask
    tokens = sample.input_ids

    res = ""

    for span, mask in zip(spans, masks):
        if mask != 1:
            continue
        res += " "
        for index in range(span[0], span[1]):

            tok = tokens[index]
            if len(span) == 4:
                color = bcolors.OKGREEN if span[3] == 1 else bcolors.FAIL
            else:
                color = bcolors.ENDC
            res += f" {color}{tok}{bcolors.ENDC}"

    return res


# collect function for batch creation for rotowire task 1
def collect2(examples: List[SpanClassificationTrainingSample],
            random_mask : Union[int, None] = None,
            random_mask_proba : Union[float, None] = 0.5) -> dict :
    """
    Collect function to form batches from sample data list.
    :param examples: List of training example
    :return: batch dictionary
    """

    input_ids: List[List[int]] = []
    masks: List[List[int]] = []
    labels: List[List[int]] = []
    weights: List[float] = []
    spans: List[List[spanType]] = []
    text_list: List[str] = []
    mentions_list: List[dict] = []

    for ex in examples:



        if random_mask_proba is not None :
            ex.input_ids = random_masking(random_mask,
                                         ex.input_ids,
                                         ex.spanList,
                                         random_mask_proba)

        input_ids.append(ex.input_ids)
        masks.append(ex.attention_mask)
        labels += ex.labels
        weights += ex.weights
        spans.append(ex.spanList)
        text_list.append(ex.text)
        mentions_list.append(ex.mentions)


    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(masks)
    labels = torch.LongTensor(labels)
    weights = torch.Tensor(weights)
    spans = spans

    return {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "labels" : labels,
        "weights" : weights,
        "spans" : spans,
        "texts" : text_list,
        "mentions_list" : mentions_list
    }

def pretty_print(token_ids, labels, tokenizer) :

    res = []

    tokens = tokenizer.decode(token_ids, skip_special_tokens = True)
    tokens = tokenizer(tokens).tokens()
    for token, label in zip(tokens, labels) :

        if token == "[PAD]" :
            break

        if label == 1 :
            token = f"\033[92m{token}\033[0m"
        elif label == 2 :
            token = f"\033[91m{token}\033[0m"

        res.append(token)

    print(" ".join(res))

# collect function for batch creation for rotowire task 2 Joint
def collect_task_two_joint(examples: List[SequenceTaggingTrainingSample],
                           onfly_preprocessor : Optional[OnFlyPreprocessor] = None) -> dict :
    """
    Collect function to form batches from sample data list.
    :param examples: List of training example
    :return: batch dictionary
    """

    input_ids: List[List[int]] = []
    masks: List[List[int]] = []
    labels: List[List[int]] = []
    weights: List[float] = []
    text_list: List[str] = []
    mentions: List[List[dict]] = []
    spanList = []
    labels_mask = []
    for ex in examples :



        if onfly_preprocessor :
            print(build_string_from_spans(ex))
            ex = onfly_preprocessor.apply(ex)

            print(build_string_from_spans(ex))
            print(ex)
            exit()

        input_ids.append(ex.input_ids)
        masks.append(ex.attention_mask)
        labels.append(ex.labels_sequence)
        weights.append(ex.weights_sequence)
        text_list.append(ex.text)
        mentions.append(ex.mentions)
        spanList.append(ex.span_list)
        labels_mask.append(ex.label_mask)

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(masks)
    labels = torch.LongTensor(labels)
    weights = torch.Tensor(weights)
    labels_mask = torch.Tensor(labels_mask)

    return {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "labels" : labels,
        "weights" : weights,
        "texts" : text_list,
        "mentions" : mentions,
        "span_list" : spanList,
        "label_mask" : labels_mask
    }


# collect function for batch creation for rotowire task 2 Joint
def collect(examples: List[DataSample],
            onfly_preprocessor : Optional[OnFlyPreprocessor] = None) -> dict :
    """
    Collect function to form batches from sample data list.
    :param examples: List of training example
    :return: batch dictionary
    """

    if onfly_preprocessor :
        examples = [onfly_preprocessor.apply(ex) for ex in examples]

    batch = DataSample.merge_samples(examples)

    return batch



