import torch
import torch.nn
from allennlp.modules.elmo import Elmo, batch_to_ids
from source.utils.register import register

@register('MODULES')
class ElmoEmbedding(torch.nn.Module) :
    def __init__(self,
                 elmo_options_file,
                 elmo_weight_file,
                 input_key : str = "input_ids",
                 output_key : str = "elmo_embedding",
                 freeze_weigths : bool = False,
                 require_grad : bool = True) :

        assert elmo_options_file is not None, f"No elmo option file provided"
        assert elmo_weight_file is not None, f"No elmo option file provided"

        super().__init__()

        self.elmo = Elmo(elmo_options_file, elmo_weight_file, 2, requires_grad=require_grad)

        self.input_key = input_key
        self.output_key = output_key

        self.freeze_weigths = freeze_weigths
        if freeze_weigths :
            for param in super().parameters():
                param.requires_grad = False


    def get_output_dim(self) :
        return self.elmo.get_output_dim()

    def forward(self, batch) :

        ids = batch[self.input_key]

        embeddings = self.elmo(ids)

        batch[self.output_key] = embeddings['elmo_representations'][0]
        batch['label_mask'] = embeddings['mask']

        #print(f"{batch[self.output_key].size()} {batch['label_mask'].size()}")
        #exit()

        return batch