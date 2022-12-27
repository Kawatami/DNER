from argparse import Namespace
from typing import List, Type
from argparse import ArgumentParser
from source.utils.register import Registers
from source.callbacks.callbacks import BaseCallback

# default callbacks to build in every cases
default_callback = ["EarlyStopping", "ModelCheckpoint"]

class CallbacksFactory :
    """
    Class factory to build dynamically the list of callbacks from their names
    """

    def __init__(self, args : Namespace) :
        self.callback_names = args.callbacks
        self.verbose = args.verbose
        self.args = args
        self.args.callback_verbose = args.callback_verbose
        self.no_default_callback = args.no_default_callback

    def build_callbacks(self) -> List[Type[BaseCallback]] :
        """
        Build a list of callbacks given a list of names
        :return: List of callbacks object
        """

        res = []
        defaults = [] if self.no_default_callback else default_callback

        for name in set(self.callback_names + defaults) :
            callback = Registers["CALLBACKS"][name].build_from_args(self.args)
            res.append(callback)

        return res

    @staticmethod
    def add_callbackFactory_specific_args(parent_parser : ArgumentParser) -> ArgumentParser:
        """
        Add the specific parsing parameter of the callback factory to the parser
        :param parent_parser: parser receiving the parameters
        :return: parser
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('CallbacksFactory')
        group.add_argument("--no_default_callback", action="store_true"
                           , default=False)
        return parser

    @staticmethod
    def add_default_callback_specific_args(parent_parser : ArgumentParser) -> ArgumentParser:
        """
        Add the specific parsing parameter of default callbacks to the parser
        :param parent_parser: parser receiving the parameters
        :return: parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('CallbacksFactory')

        for callback in default_callback :
            group = \
                Registers["CALLBACKS"][callback].add_callback_specific_args(group)
        return parser