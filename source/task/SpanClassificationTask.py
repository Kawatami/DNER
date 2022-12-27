from source.task.base_task import BaseTask
from source.utils.register import register
from argparse import ArgumentParser
from typing import List

@register("TASKS")
class SpanClassificationTask(BaseTask) :

    def __init__(self, lr : float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        if kwargs['dataset'] == 'ImDBDataModule':
            self.test_set_names = [
                "unseen",
                "seen",
                "all"
            ]

    @staticmethod
    def add_task_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('SpanClassificationTask')

        group.add_argument("--lr", type=float, default=1e-5)
        group.add_argument("--dataset", type=str, default='RotoWireDataModule',
                            choices=['RotoWireDataModule', 'ImDBDataModule'])
        group.add_argument("--test_set_names", nargs="+",
                           default=['seen\\unseen', 'unseen', 'seen', 'all'])
        return parser