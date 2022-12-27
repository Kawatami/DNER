from torch.utils.data import Dataset
from typing import Any, List, Callable
from source.data.utils.training_batch import  SpanClassificationTrainingSample
from source.data.data_sample.DataSample import DataSample

class BaseDataset(Dataset):
    """
    RotoWire dataset used by torch DataLoader
    """
    def __init__(self,
                 samples_creator: Callable,
                 text_preprocessing : List[Callable] = list(),
                 label_preprocessing: List[Callable] = list()):
        """
        Aply preprocessing and format data to be usable by a model
        :param tokenizer: PreTrainedTokenizer from huggingface transformer library
        :param data: List of samples
        :param text_preprocessings: list of text preprocessing
        :param label_preprocessings: list of labels preprocessing
        """
        super().__init__()
        self.text_preprocessing : List[Callable] = text_preprocessing
        self.label_preprocessing : List[Callable] = label_preprocessing
        self.processed_data : Any = None
        self.samples_creator = samples_creator

    def preprocess_data(self, data : List[dict]) -> List[dict] :
        """
        Preprocess raw data before tokenization
        :param data: List of raw data
        :return: list of raw data updated
        """

        raise NotImplementedError

    def process_data(self, data : List[dict], **kwargs) -> None :
        """
        Tokenize data and produce datasets
        :param data: lost of sample
        """
        if not data :
            return

        # preprocessing
        data = self.preprocess_data(data)

        # apply model sample creation method
        self.processed_data = self.samples_creator(data, **kwargs)


    def __len__(self):
        if self.processed_data is not None :
            return len(self.processed_data)
        else :
            raise AttributeError(
                f"Processed data list is not initialized. Have you provided "
                f"any data ?"
            )

    def __getitem__(self, item) -> DataSample :
        if self.processed_data:

            return self.processed_data[item]
        else:
            raise AttributeError(
                f"Processed data list is not initialized. Have you provided "
                f"any data ?"
            )
