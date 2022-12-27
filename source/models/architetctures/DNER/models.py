from __future__ import annotations
from source.models.architetctures.DNER.base_model import DNERBase
from source.utils.register import register, Registers
from argparse import ArgumentParser, Namespace
from typing import Any, List, Tuple, Optional
import torch
from tqdm import tqdm
from source.data.utils.training_batch import SequenceTaggingTrainingSample
from transformers import BertTokenizerFast
from tokenizers import Encoding
from source.data.data_sample.DataSample import DataSample
from functools import cached_property
from source.data.tokenizer.tokenizer import TokenizerLoader

from enum import Enum
import numpy as np

@register('MODELS')
class DNERBertLinear(DNERBase) :
    """
    Base line model for RotoWire Task 2
    """
    @classmethod
    def from_args(cls, args : Namespace) -> DNERBertLinear:
        """
        Build model from main parser namespace.
        :param args: namespace issued by main parser
        :return: model
        """

        context_encoder = Registers['MODULES']["BERT"](
            training_key = args.training_key,
            position_type = args.position_type
        )

        word_pooling = Registers['MODULES']["MaxPoolingWord"](
            input_name = "transformer_hidden_states",
            output_name = "pooled_vector",
            dim = 1,
            concat_CLS = False
        )

        classifier = Registers['MODULES']["Linear"](
            in_features = context_encoder.config.hidden_size,
            out_features = args.out_features,
            bias = args.bias,
            input_name = "pooled_vector",
            output_name = args.out_name
        )

        return cls(context_encoder=context_encoder,
                   classifier=classifier,
                   word_pooling=word_pooling)

    @staticmethod
    def add_model_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :
        """
        Add model specific args to main parser
        :param parent_parser: main parser
        :return: main parser updated
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Task 1 Baseline')

        # Embedding arguments
        group.add_argument('--training_key', type=str, default="bert-base-cased")
        group.add_argument('--out_features', type=int, default=9)
        group.add_argument('--position_type', type=str, default="absolute",
                           choices=["absolute", "relative_key", "relative_key_query"])

        group.add_argument('--bias', type=bool, default=True)


        # Pooling arguments
        group.add_argument('--out_name', type=str, default="prediction")
        return parser

    @cached_property
    def model_tokenizer(self):
        return TokenizerLoader().load_tokenizer(BertTokenizerFast, 'bert-base-cased')

    def create_samples(self, raw_samples : Any, **kwargs) :
        """
        Generate the preprocessed samples
        """

        # getting weight and padding info
        weights = kwargs['weights']
        padding = kwargs['padding']


        # loading tokenizer
        tokenizer = self.model_tokenizer

        # tokenzie texts
        texts = [sample['preprocessed_text'] for sample in raw_samples]
        tokenized_batch = tokenizer(texts,
                                    return_offsets_mapping=True,
                                    padding="max_length",
                                    truncation=True)

        processed_data = []

        # iterating over samples
        for index, sample in tqdm(enumerate(raw_samples)) :
            tokenized_text: Encoding = tokenized_batch[index]

            # get spans token wise
            spans = []
            labels = []

            # iterating over entities
            for entity in sample['entities']:

                # getting indexes BPE wise of the entity from its char indexes
                token_span = self.convert_charspan_to_tokenspan(tokenized_text, entity['span'])
                entity['token_span'] = token_span

                # if entity not ouside the processing window
                if token_span is not None :
                    spans.append(token_span)
                    labels.append(entity['label'] * 4 + 1) # updating labels to follow IOBES scheme

            # extract word span
            word_spans = self.extract_word_spans(tokenized_text.word_ids)


            # merging entity spans
            word_spans = self.update_ground_truth_entities(word_spans, spans, labels)

            #print(word_spans)
            #exit()
            # generate training data
            label_sequence, weight_sequence, mask = self.generate_ground_truth(word_spans, weights, padding=padding)

            # if no spans are with the tokens limit ignore sample
            if spans == []:
                continue

            gpu = {
                "input_ids": tokenized_text.ids,
                "attention_mask": tokenized_text.attention_mask,
                "labels": label_sequence,
                "weights": weight_sequence,
                "labels_mask": mask
            }

            cpu = {
                "spans": word_spans,
                "text": tokenized_text.tokens,
                "entities": sample['entities']
            }

            sample = DataSample(gpu=gpu, cpu=cpu)

            #print(sample)
            #exit()

            processed_data.append(sample)

        return processed_data

    def build_string_from_spans(self, spans, tokens, masks):

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



        res = ""

        for span, mask in zip(spans, masks):
            if mask != 1:
                continue
            res += " "
            for index in range(span[0], span[1]):

                tok = tokens[index].replace("##", "")
                if len(span) == 4:
                    color = bcolors.OKGREEN if span[3] == 1 else bcolors.FAIL
                else:
                    color = bcolors.ENDC
                res += f"{color}{tok}{bcolors.ENDC}"

        return res

    def extract_word_spans(self, word_ids) :
        """
        Process the span for each word
        """

        ref = word_ids[0]
        begin = 0
        res = []
        first = True
        for i, ids in enumerate(word_ids):
            if ref != ids:
                res.append((begin, i, 1))
                ref = ids
                begin = i

            if ids is None:
                if first:
                    first = False
                    continue
                else:
                    res.append((begin, i, 0))
                    ref = ids
                    begin = i
        return res


    def convert_charspan_to_tokenspan(
            self,
            tokenized_text: Encoding,
            span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Convert a char base representation to its corresponding token based
        span representation
        :param tokenized_text: Object issued by the tokenizer
        :param span: char based span tuple
        :return: token based span representation
        """
        indexes = set()
        for char_idx in range(span[0], span[1]):
            token_idx = tokenized_text.char_to_token(char_idx)
            if token_idx is not None:
                indexes.add(token_idx)

        indexes = list(indexes)
        indexes.sort()

        if len(indexes):
            return indexes[0], indexes[-1]
        else:
            return None

    def update_ground_truth(self,
                            e_span : Tuple[int, int],
                            spans : List[Tuple[int, int, int]],
                            label : int) :
        """
        Add label to word spans when they correspond to an entity.
        Label are organized as follows :

        class number : 0   1   2   3   4   5   6   7   8
        meaning      : 0   BL  IL  EL  SL  BW  IW  EW  SW

        BL : Begin Loser
        IL : Inside Loser
        EL : End Loser
        BW : Begin Winner
        IW : Inside Winner
        EW : End Winner
        SW : Single Winner
        0  : Not an entity

        """

        class States(Enum):
            FOUND = 0,
            SEARCHING = 1,
            FLUSH = 2

        # defining entity span boundaries
        begin = e_span[0]
        end = e_span[1]

        # flag indicating when the entity is found
        found = False

        state = States.SEARCHING

        res = []
        for span in spans:


            if state == States.FLUSH :
                res.append(span)
            elif state == States.SEARCHING :
                if span[0] == begin and span[1] == end + 1 : # Found SINGLE
                    res.append((span[0], span[1], span[2], label + 3))
                    state = States.FLUSH
                elif span[0] == begin and span[1] != end + 1 : # found BEGIN
                    res.append((span[0], span[1], span[2], label))
                    state = States.FOUND
                else :
                    res.append(span)
            elif state == States.FOUND :
                # if the current index is the end of the entity restart the flag
                if span[1] == end + 1:
                    current_label = label + 2  # current label -> end
                    state = States.FLUSH
                else:
                    current_label = label + 1  # current label -> inside

                res.append((span[0], span[1], span[2], current_label))  # push it to the output

        return res

    def update_ground_truth_entities(self, spans, entities_spans, labels):

        for e_span, label in zip(entities_spans, labels):
            spans = self.update_ground_truth(e_span, spans, label)

        return spans

    def generate_ground_truth(self, span_list, weights, padding : int = 512) :

        def pad(sequence) :
            return np.pad(sequence, (0, padding - len(sequence))).tolist()

        def get_associated_class(label) :
            """
            Return the associated label given a sub label (B, I, E, S)
            + label - 1 to not take into account the "not an entity" class
            + % 4 because there are 4 sub classes (B, I, E, S)
            + // 4 to get back to the label of BEGIN class in every cases
            + + 1 to take into account the "not an entity" class
            """
            return (label - (label - 1) % 4) // 4 + 1

        res_label = []
        res_weights = []
        res_mask = []
        for span in span_list:
            if len(span) == 3:
                res_label.append(0)
                res_weights.append(weights[0])
            else:
                res_label.append(span[3]) # adding + 1 as non entity token use label 0
                res_weights.append(weights[get_associated_class(span[3])])
            res_mask.append(span[2])

        res_mask[0] = 0


        return pad(res_label), pad(res_weights), pad(res_mask)

@register('MODELS')
class DNERBertLinearCLS(DNERBertLinear):
    """
    Base line model for RotoWire Task 2
    """

    @classmethod
    def from_args(cls, args: Namespace) -> DNERBertLinearCLS:
        """
        Build model from main parser namespace.
        :param args: namespace issued by main parser
        :return: model
        """

        context_encoder = Registers['MODULES']["BERT"](
            training_key=args.training_key
        )

        word_pooling = Registers['MODULES']["MaxPoolingWord"](
            input_name="transformer_hidden_states",
            output_name="pooled_vector",
            dim=1,
            concat_CLS=True
        )

        classifier = Registers['MODULES']["Linear"](
            in_features=context_encoder.config.hidden_size * 2,
            out_features=args.out_features,
            bias=args.bias,
            input_name="pooled_vector",
            output_name=args.out_name
        )

        return cls(context_encoder=context_encoder,
                   classifier=classifier,
                   word_pooling=word_pooling)



