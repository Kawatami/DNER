from torch.utils.data import Dataset
from tokenizers import Encoding
from typing import Any, List, Callable, Tuple, Type, Union
from source.data.utils.training_batch import  SpanClassificationTrainingSample
from tqdm import tqdm
from source.data.datasets.BaseDataset import BaseDataset

class DNERDataset(BaseDataset):
    """
    RotoWire dataset used by torch DataLoader
    """
    def __init__(self,
                 samples_creator: Callable,
                 text_preprocessing : List[Callable] = list(),
                 label_preprocessing: List[Callable] = list()):

        super().__init__(
            samples_creator=samples_creator,
            text_preprocessing=text_preprocessing,
            label_preprocessing=label_preprocessing
        )

    def preprocess_data(self, data : List[dict]) -> List[dict] :
        """
        Preprocess raw data before tokenization
        :param data: List of raw data
        :return: list of raw data updated
        """

        # retrieving data
        texts = [game['text'] for game in data]
        labels = [game['entities'] for game in data]

        # apply preprocessings
        for preprocessing in self.text_preprocessing:
            texts = list(map(preprocessing, texts))

        for preprocessing in self.label_preprocessing:
            labels = list(map(preprocessing, labels))

        data_res = []
        # storing info
        for i, (label, text, sample) in enumerate(zip(labels, texts, data)):
            sample['preprocessed_text'] = text
            sample['preprocessed_labels'] = label
            data_res.append(sample)


        return data_res
