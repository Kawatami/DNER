from transformers import BertTokenizerFast,  LongformerTokenizerFast
from tokenizers import Encoding
from transformers import  BertModel
import pathlib
import logging
import argparse
from transformers import AutoTokenizer

def resolve_tokenizer(tokenizer : str) :
    return tokenizers[tokenizer]


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class TokenizerLoader(metaclass=Singleton) :

    def __init__(self, force_local_load : bool = False, **kwargs):

        self.force_local_load = force_local_load

    @staticmethod
    def add_tokenizer_loader_specific_args(parent_parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        group = parser.add_argument_group("TokenizerLoader")
        group.add_argument("--force_local_load", action="store_true", default=False)

        return parser

    def load_tokenizer(self, tokenizer_class, key : str, **tokenizer_param) :


            # Loading from web
            if not self.force_local_load :
                logging.info(f"Attempting tokenizer \"{key}\" from web.")

                try :
                    tokenizer = tokenizer_class.from_pretrained(key, **tokenizer_param)
                    logging.info(f"Successfully loaded tokenizer {key}")
                    return tokenizer

                except ValueError :
                    logging.error(f"Attempt failed trying local file")

            else :
                logging.info("force_local_load enabled, only loading from local file.")

            # loading from local files
            logging.info(f"Loading tokenizer \"{key}\" from local files.")

            path_ressources = self.get_tokenizer_path(tokenizer_class.__name__)
            tokenizer = tokenizer_class.from_pretrained(path_ressources, **tokenizer_param)

            return tokenizer

    def get_tokenizer_path(self, key : str) -> pathlib.Path :
        """
        Provides path for tokenizer files
        """

        if key not in tokenizers.keys() :
            raise ValueError(f"tokenizer \"{key}\" not supported.")
        else :
            return tokenizers[key]



path_tokenizer_ressources = pathlib.Path(__file__).parents[0] / "tokenizer_files"

tokenizers = {
    "BartTokenizerFast" : path_tokenizer_ressources / "BartTokenizerFast",
    "BertTokenizerFast": path_tokenizer_ressources / "BertTokenizerFast",

    #"LongFormerTokenizerFast" : LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096',proxies=proxy)
}