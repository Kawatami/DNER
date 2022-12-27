from __future__ import annotations
from source.models.architetctures.DNET.base_model import DNETBase
from source.utils.register import register, Registers
from argparse import ArgumentParser, Namespace
from transformers import BertTokenizerFast, PreTrainedTokenizer
from tqdm import tqdm
from typing import Any, List, Tuple, Optional
from tokenizers import Encoding
from source.data.data_sample.DataSample import DataSample

from source.utils.misc import MODEL_FILES_ROOT

@register('MODELS')
class BertLinear(DNETBase) :
    """
    BertLinear Architecture
    """
    @classmethod
    def from_args(cls, args : Namespace) -> BertLinear:
        """
        Build model from main parser namespace.
        :param args: namespace issued by main parser
        :return: model
        """

        context_encoder = Registers['MODULES']["BERT"](
            training_key = args.training_key,
            position_type = args.pos_type
        )

        span_encoder = Registers['MODULES']["MaxPoolingSequenceSpans"](
            input_name = "transformer_hidden_states",
            output_name = "pooled_vector"
        )

        classifier = Registers['MODULES']["Linear"](
            in_features = context_encoder.config.hidden_size,
            out_features = args.out_features,
            bias = args.bias,
            input_name = "pooled_vector",
            output_name = args.out_name
        )

        return cls(context_encoder=context_encoder,
                   span_encoder=span_encoder,
                   classifier=classifier,
                   last_activation=args.last_activation)

    def create_samples(self, raw_samples : Any, **kwarg) :


        if hasattr(self, "tokenizer") :
            tokenizer = self.tokenizer
        else :
            self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_FILES_ROOT / self.context_encoder.training_key)
            tokenizer = self.tokenizer

        weights = kwarg['weights']


        tokenized_text = tokenizer([sample['preprocessed_text'] for sample in raw_samples],
                                   padding="max_length",
                                   truncation=True,
                                   max_length=512)

        processed_data = list()
        for index, raw_sample in tqdm(enumerate(raw_samples)):
            tokens : Encoding = tokenized_text[index]

            entities = raw_sample['entities']

            entities.sort(key = lambda x : x['span'][0])

            # extract all spans and labels
            spans, labels, labels_weights, labels_mask = self.preprocess_spans_labels(
                entities,
                tokens,
                weights
            )

            # if no spans are with the tokens limit ignore sample
            if spans == [] :
                continue

            gpu = {
                "input_ids" : tokens.ids,
                "attention_mask":tokens.attention_mask,
                "labels" :labels,
                "weights" : labels_weights,
                "labels_mask" : labels_mask
            }

            cpu = {
                "spans": spans,
                "text": tokens.tokens,
                "entities": entities
            }

            sample = DataSample(gpu=gpu, cpu=cpu)

            processed_data.append(sample)

        return processed_data

    def preprocess_spans_labels(self,
                                entities,
                                tokenized_text : Encoding,
                                weights : List[float],
                                sequence_length : int = 100) -> Tuple[List[Tuple[int, int]],List[int], List[float], List[int]]:
        """
        Convert char based span to token based spans. Preprocess labels to keep
        only the ones within the max sequence length accepted by the tokenizer
        :param spans:
        :param labels:
        :param tokenized_text:
        :return:
        """

        res_spans = []
        res_label = [0] * sequence_length
        res_weight = [0.0] * sequence_length
        res_mask_label = [0] * sequence_length

        for index, entity in enumerate(entities) :

            # convert span and storing
            span = self.convert_charspan_to_tokenspan(tokenized_text, entity['span'])

            # if span is out of bound stop
            if span is None :
                continue

            res_spans.append(span)

            # storing corresponding label
            res_label[index] = entity['label']
            res_mask_label[index] = 1

            # storing corresponding weight
            res_weight[index] = weights[entity['label']]

        return res_spans, res_label, res_weight, res_mask_label

    def convert_charspan_to_tokenspan(self,
                                      tokenized_text : Encoding,
                                      span : Tuple[int, int]) -> Optional[Tuple[int, int]] :
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

        if len(indexes) :
            return indexes[0], indexes[-1]
        else :
            return None


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
        group.add_argument('-training_key', type=str, default="bert-base-cased")
        group.add_argument('--out_features', type=int, default=1)
        group.add_argument('--bias', type=bool, default=True)
        group.add_argument('--last_activation', type=str, default="sigmoid")
        group.add_argument('--pos_type', type=str, default="absolute", choices=["absolute", "relative_key"])


        # Pooling arguments
        group.add_argument('--out_name', type=str, default="prediction")
        return parser

@register("MODELS")
class BertLinearCLS(BertLinear) :
    """
    BertLinear CLS architecture
    """

    @classmethod
    def from_args(cls, args: Namespace) -> BertLinearCLS:
        """
        Build model from main parser namespace.
        :param args: namespace issued by main parser
        :return: model
        """

        context_encoder = Registers['MODULES']["BERT"](
            training_key=args.training_key
        )

        span_encoder = Registers['MODULES']["MaxPoolingSequenceSpans"](
            input_name="transformer_hidden_states",
            output_name="pooled_vector",
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
                   span_encoder=span_encoder,
                   classifier=classifier,
                   last_activation=args.last_activation)



