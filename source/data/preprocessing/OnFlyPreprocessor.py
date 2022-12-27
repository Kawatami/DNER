from source.data.preprocessing.onfly_preprocessing import onfly_preprocessing
from argparse import ArgumentParser

class OnFlyPreprocessor :

    def __init__(self, list_preprocessing) :

        # checking preprocessing
        error_preproc = [
            preproc for preproc in list_preprocessing
            if preproc not in onfly_preprocessing.keys()
        ]

        if error_preproc != [] :
            raise ValueError(f"Unrecognized onfly preprocessing {error_preproc}."
                             f"Available onfly preprocessing : {onfly_preprocessing.keys}")
        self.name_preproc = list_preprocessing
        self.list_preproc = [
            onfly_preprocessing[preproc] for preproc in list_preprocessing
        ]

    def __repr__(self) :
        return "Onfly preprocessing :\n" + "".join([f"+ {preproc}\n" for preproc in self.name_preproc])

    @classmethod
    def build_from_args(cls, args) :
        return cls(list_preprocessing=args.list_onfly_preprocessing)

    @staticmethod
    def add_onfly_preprocessor_args(parent_parser : ArgumentParser) -> ArgumentParser :
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Onfly arguments')
        group.add_argument('--list_onfly_preprocessing', nargs="+", default=[],
                           choices=onfly_preprocessing.keys())
        return parser

    def apply(self, example) :
        for preproc in self.list_preproc :
            example = preproc(example)
        return example

