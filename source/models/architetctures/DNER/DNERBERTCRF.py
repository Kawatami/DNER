from __future__ import annotations
from source.utils.register import register, Registers
from argparse import ArgumentParser, Namespace
from source.models.architetctures.DNER.models import DNERBertLinear

@register('MODELS')
class DNERBERTCRF(DNERBertLinear) :
    """
    Base line model for RotoWire Task 2
    """
    @classmethod
    def from_args(cls, args : Namespace) -> DNERBERTCRF:
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
            input_name="transformer_hidden_states",
            output_name="pooled_vector",
            dim=1,
            concat_CLS=False
        )

        classifier = Registers['MODULES']["CRFLayer"](
            hidden_features = context_encoder.config.hidden_size,
            out_features = args.out_features,
            input_name = "pooled_vector",
            output_name = args.out_name,
            labels_name = args.labels_name
        )

        return cls(context_encoder=context_encoder,
                   classifier=classifier,
                   word_pooling=word_pooling)

    def forward(self, batch: dict):
        """
        inference method
        :param batch: batch of data
        :return: batch of data updated with intermediary states
        """
        # context encoding
        batch = self.context_encoder(batch)

        if self.word_pooling is not None:
            # pooling to word token
            batch = self.word_pooling(batch)

        # classification
        batch = self.classifier(batch)
        batch['prediction'] = batch['prediction'].permute(0, 2, 1)

        return batch

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
        group.add_argument("--dropout", type=float, default=0.0)
        group.add_argument('--bias', type=bool, default=True)


        # Pooling arguments
        group.add_argument('--out_name', type=str, default="prediction")
        group.add_argument('--input_name', type=str, default="transformer_hidden_states")
        group.add_argument('--labels_name', type=str, default="labels")

        return parser

@register('MODELS')
class DNERBERTCRFCLS(DNERBERTCRF) :
    """
    Base line model for RotoWire Task 2
    """
    @classmethod
    def from_args(cls, args : Namespace) -> DNERBERTCRFCLS:
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
            input_name="transformer_hidden_states",
            output_name="pooled_vector",
            dim=1,
            concat_CLS=True
        )

        classifier = Registers['MODULES']["CRFLayer"](
            hidden_features = context_encoder.config.hidden_size * 2,
            out_features = args.out_features,
            input_name = "pooled_vector",
            output_name = args.out_name,
            labels_name = args.labels_name
        )

        return cls(context_encoder=context_encoder,
                   classifier=classifier,
                   word_pooling=word_pooling)

