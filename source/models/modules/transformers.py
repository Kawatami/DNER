from __future__ import annotations
from typing import Type, Union
from argparse import Namespace
from transformers import PreTrainedModel, BertModel, LongformerModel, AutoConfig, AutoModel
from transformers import BartModel
from source.utils.register import register
import torch
import pathlib
from source.utils.misc import MODEL_FILES_ROOT, proxy
import logging
from ..modules.CustomBert.CustomBert import CustomAnchorBert
from ..modules.CustomBert.CustomBert import CustomAnchorBert

class TransformerBaseWrapper(torch.nn.Module) :
    """
    Wrapper for transformer architecture from huggingface library
    """
    def __init__(self,
                 model : Type[PreTrainedModel],
                 training_key : str,
                 output_key_name : str = "transformer") :
        """
        Transformer Base Wrapper constructor
        :param model: PretrainedModel from huggingface
        :param training_key: pre training key for weights loading
        :param output_key_name: okey to use for output storing
        """
        super().__init__()

        self.model = model
        self.output_key_name = output_key_name
        self.training_key = training_key

    @classmethod
    def from_args(cls, args : Namespace) -> TransformerBaseWrapper:
        return cls(args.transformer,
                   args.transformer_pretraining_key,
                   args.transformer_output_key)

    @property
    def config (self) -> PreTrainedModel.config_class :
        """
        Getter for model configuration object
        :return: config object
        """
        return self.model.config


    def forward(self, batch : dict) -> dict :


        outputs = self.model(input_ids = batch['input_ids'],
                             attention_mask = batch['attention_mask'])

        last_hidden_states = outputs.last_hidden_state

        batch[f'{self.output_key_name}_output'] = outputs
        batch[f'{self.output_key_name}_hidden_states'] = last_hidden_states
        batch[f'CLS_hidden_states'] = last_hidden_states[:, 0, :]

        return batch



@register('MODULES')
class AnchorBERT(TransformerBaseWrapper) :
    def __init__(self,
                 training_key: str,
                 output_key_name: str = "transformer",
                 anchor_points : int = 10,
                 pos_type : str = "fixed",
                 sharpness_pos : float = 1.0,
                 base_pos_type : str = "absolute",
                 use_base_pos_embedding : bool = False,
                 hidden_size_pos_emb : int = 15,
                 multihead_drop_out : float = 0.3,
                 pos_emb_num_heads : int = 1
                 ):
        """
        Transformer Base Wrapper constructor
        :param model: PretrainedModel from huggingface
        :param training_key: pre training key for weights loading
        :param output_key_name: okey to use for output storing
        """

        if not use_base_pos_embedding :
            base_pos_type = "absolute"

        config = AutoConfig.from_pretrained(training_key, position_embedding_type=base_pos_type)
        config.anchor_points = anchor_points
        config.pos_type = pos_type
        config.sharpness_pos = sharpness_pos
        config.use_base_pos_embedding = use_base_pos_embedding
        config.hidden_size_pos_emb = hidden_size_pos_emb
        config.multihead_drop_out = multihead_drop_out
        config.pos_emb_num_heads = pos_emb_num_heads
        logging.info(f"Loading model from web")
        model = CustomAnchorBert.from_pretrained(training_key, config=config)

        super().__init__(
            model=model,
            training_key=training_key,
            output_key_name=output_key_name
        )


@register('MODULES')
class BERT(TransformerBaseWrapper) :
    """
    BERT Model from huggingface
    """
    def __init__(self,
                 training_key : str = "bert-base-uncased",
                 output_key_name : str = "transformer",
                 position_type : str = "absolute"):

        model = BertModel.from_pretrained(
            training_key,
            position_embedding_type=position_type
        )

        super().__init__(
            model=model,
            training_key=training_key,
            output_key_name=output_key_name
        )

@register('MODULES')
class BART(TransformerBaseWrapper) :
    """
    BERT Model from huggingface
    """
    def __init__(self,
                 training_key : str = 'facebook/bart-large',
                 output_key_name : str = "transformer"):

        model = BartModel.from_pretrained(
            training_key,
        )

        super().__init__(
            model=model,
            training_key=training_key,
            output_key_name=output_key_name
        )


# TODO : Recode
@register('MODULES')
class BERTVanilla(TransformerBaseWrapper) :
    """
    BERT Model from huggingface not pretrained
    """
    def __init__(self,
                 training_key : str = "bert-base-cased",
                 output_key_name : str = "transformer"):
        super().__init__(AutoConfig, training_key, output_key_name)

        self.model = AutoModel.from_config(self.model)

# TODO : Recode
@register('MODULES')
class LongFormer(TransformerBaseWrapper) :
    """
    LongFormer Model from huggingface
    """
    def __init__(self,
                 training_key : str = "allenai/longformer-base-4096",
                 output_key_name : str = "transformer"):
        super().__init__(LongformerModel, training_key, output_key_name)