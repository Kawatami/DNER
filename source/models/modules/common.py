from typing import Union
from source.utils.register import register
import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF
from source.models.modules.CustomBert.CustomEmbedding import AnchorsBertEmbeddings
@register('MODULES')
class Linear(torch.nn.Module):
    """
    Simple Linear modules, bottled in a class to have the correct API.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 input_name: str = 'context',
                 output_name: str = 'prediction'):
        """
        Linear operation constructor.

        :in_features: dimension of input feature
        :out_features: dimension of output feature
        :bias: to use bias or not
        :output_name: Name of the key in forward returned dict.
        """
        super().__init__()

        self.input_name = input_name
        self.output_name = output_name
        self.linear = torch.nn.Linear(in_features, out_features, bias)

    def forward(self, batch : dict) -> dict:
        batch[self.output_name] = self.linear(batch[self.input_name])
        return batch

@register('MODULES')
class FullyConnected(torch.nn.Module) :
    """
    Fully connected modules, bottled in a class to have the correct API.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features : int,
                 n_layers : int,
                 activation : str = "ReLu",
                 bias: bool = True,
                 input_name: str = 'context',
                 output_name: str = 'prediction'):
        """
        Linear operation constructor.

        :in_features: dimension of input feature
        :out_features: dimension of output feature
        :bias: to use bias or not
        :output_name: Name of the key in forward returned dict.
        """
        super().__init__()

        self.input_name = input_name
        self.output_name = output_name


        self.net = [torch.nn.Linear(in_features, hidden_features, bias),
                    torch.nn.ReLU()]

        for l in range(n_layers - 2) :
            self.net += [torch.nn.Linear(hidden_features, hidden_features, bias),
                    torch.nn.ReLU()]

        self.net += [torch.nn.Linear(hidden_features, out_features, bias)]

        self.net = torch.nn.Sequential(*self.net)

    def forward(self, batch: dict) -> dict:
        batch[self.output_name] = self.net(batch[self.input_name])
        return batch

@register('MODULES')
class ParallelLinear(torch.nn.Module):
    """
    Same class as Linear, but can be applied to several keys
    """
    __default_input_names = ['context', 'raw_encoded_values']
    __default_output_names = ['prediction', 'ae_prediction']

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 input_names: Union[str] = None,
                 output_names:  Union[str] = None):
        super().__init__()

        self.input_name = input_names or self.__default_input_names
        self.output_name = output_names or self.__default_output_names
        self.linear = torch.nn.Linear(in_features, out_features, bias)

    def forward(self, model_inputs : dict) -> dict:
        for in_key, out_key in zip(self.input_names, self.output_names):
            if out_key in model_inputs:
                raise RuntimeError(f'Never modify an existing key! ({out_key})')
            model_inputs[out_key] = self.linear(model_inputs[in_key])
        return model_inputs

@register('MODULES')
class MaxPoolingSequence(torch.nn.Module) :
    """
    Apply Max Pooling over a sequence
    """
    def __init__(self,
                 input_name: str = "context",
                 output_name:  str = "pooled_vector",
                 dim : int  = 1):
        """
        Max Pooling sequence constructor
        :param input_name: key name to use for input batch dict
        :param output_name: key name to store output
        :param dim: dim to max pool along
        """
        super().__init__()

        self.input_name = input_name
        self.output_name = output_name
        assert 0 <= dim
        self.dim = dim

    def forward(self, batch : dict) -> dict :
        assert self.dim <= len(batch[self.input_name].shape)
        batch[self.output_name] = batch[self.input_name].max(dim=self.dim)[0]
        return batch

@register("MODULES")
class MaxPoolingSequenceSpans(torch.nn.Module) :
    """
    Apply max pooling of subsequences extracted from spans
    """
    def __init__(self,
                 input_name: str = "context",
                 output_name:  str = "pooled_vector",
                 output_shape: int = 100,
                 dim : int  = 1,
                 concat_CLS : bool = False):
        """
        Max Pooling sequence spans constructor

        :param input_name: key name to use for input batch dict
        :param output_name: key name to store output
        :param dim: dim to max pool along
        """
        super().__init__()

        self.input_name = input_name
        self.output_name = output_name
        assert 0 <= dim
        self.dim = dim
        self.concat_CLS = concat_CLS

        assert output_shape > 0
        self.output_shape = output_shape


        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else -1
        self.device_index = -1 if self.device_count <= 1 else torch.cuda.current_device()



    def forward(self, batch : dict) -> dict :

        # making sure the dimension to apply the max pooling is correct
        assert self.dim <= len(batch[self.input_name].shape) - 1

        # getting the tensor dimensions
        dims = batch[self.input_name].size()
        # getting the number of span
        num_spans = len(sum(batch['spans'], []))
        # getting the right output dimension
        hidden_state_dim = dims[-1] if not self.concat_CLS else dims[-1] * 2

        # allocating output matrix
        #res = torch.zeros(num_spans, hidden_state_dim, device=batch[self.input_name].device)
        # (batch_size, output_shape, hidden_dim)
        res = torch.zeros(dims[0], self.output_shape, hidden_state_dim, device=batch[self.input_name].device)

        for batch_index, spans in enumerate(batch['spans']) :
            for span_index, (begin, end) in enumerate(spans) :

                if begin == end :
                    pooled_vector = batch[self.input_name][batch_index, begin, :]
                else :
                    pooled_vector = batch[self.input_name][batch_index, begin:end, :].max(self.dim - 1)[0]

                res[batch_index, span_index] = pooled_vector if not self.concat_CLS \
                    else torch.cat((batch['CLS_hidden_states'][batch_index], pooled_vector), 0)

        batch[self.output_name] = res

        return batch

@register("MODULES")
class MaxPoolingWord(torch.nn.Module) :
    """
    Apply max pooling of subsequences extracted from spans
    """
    def __init__(self,
                 input_name: str = "context",
                 output_name:  str = "pooled_vector",
                 dim : int  = 1,
                 concat_CLS : bool = False):
        """
        Max Pooling sequence spans constructor

        :param input_name: key name to use for input batch dict
        :param output_name: key name to store output
        :param dim: dim to max pool along
        """
        super().__init__()

        self.input_name = input_name
        self.output_name = output_name
        assert 0 <= dim
        self.dim = dim
        self.concat_CLS = concat_CLS

    def forward(self, batch : dict) -> dict :

        # making sure the dimension to apply the max pooling is correct
        assert self.dim <= len(batch[self.input_name].shape) - 1

        # getting the tensor dimensions
        dims = list(batch[self.input_name].size())

        # doubling last dimention in case of cls concat
        if self.concat_CLS :
            dims[-1] = dims[-1] * 2

        input = batch[self.input_name]

        res = torch.zeros(dims, device = batch[self.input_name].device)

        spans = batch['spans']


        for index_batch in range(dims[0]) :
            for index_span, span in enumerate(spans[index_batch]) :

                if span[0] == span[1] :
                    continue

                # mox pooling over word sub tokens
                pooled_vector = input[index_batch, span[0]: span[1]]

                if span[1] - span[0] > 1 :
                    pooled_vector = pooled_vector.max(dim = self.dim - 1)[0]

                else :
                    pooled_vector = pooled_vector.squeeze(0)


                pooled_vector = pooled_vector if not self.concat_CLS \
                    else torch.cat((batch['CLS_hidden_states'][index_batch], pooled_vector), 0)

                #print(pooled_vector.size())
                #print(res.size())
                #exit()

                res[index_batch, index_span] = pooled_vector

        batch[self.output_name] = res

        return batch

@register("MODULES")
class LSTM(torch.nn.Module) :
    """
    BiLSTM wrapper
    """

    _init_states = [
        "CLS",
        "zeros",
        "learned"
    ]

    def __init__(self,
                 input_name : str,
                 output_name : str,
                 input_size : int = 768,
                 hidden_dim : int = 512,
                 bidirectional : bool = True,
                 n_layers : int = 1,
                 dropout : float = 0.0,
                 init_state : str = "CLS",
                 batch_size : int = 32,
                 sequence_len : int = 512) :

        super().__init__()

        if init_state not in self._init_states :
            raise ValueError(
                f"LSTM init error : init state \"{init_state}\" not supported. "
                f"Available : {self._init_states}"
            )

        assert batch_size > 0

        self.input_name     = input_name
        self.output_name    = output_name
        self.sequence_len   = sequence_len
        self.init_state     = init_state
        self.hidden_dim     = hidden_dim
        self.batch_size     = batch_size
        self.n_layer        = n_layers
        self.input_size     = input_size
        self.bidirectional  = bidirectional
        #self.proj_size      = proj_size     # TODO : fix softmax when proj_size < 0
        self.h0 = None
        self.c0 = None

        self.lstm = torch.nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_dim,
            bidirectional = bidirectional,
            batch_first = True,
            dropout = dropout,
            num_layers = n_layers,
            #proj_size = self.proj_size
        )

    def get_init_states(self, batch, batch_size) :
        """
        Called in the forward function. Initialize the hidden and cell state for the lstm.
        """
        device = batch[self.input_name].get_device()

        if self.init_state == "zeros" :
            if self.h0 is None :
                self.h0 = torch.zeros(self.n_layer * (1 + self.bidirectional), batch_size, self.hidden_dim,
                                      device = device)
            if self.c0 is None :
                self.c0 = torch.zeros(self.n_layer * (1 + self.bidirectional), batch_size, self.hidden_dim,
                                      device = device)

        if self.init_state == "learnable" :
            if self.h0 is None :
                h0 = torch.nn.init.xavier_uniform(
                    torch.zeros(self.n_layer * (1 + self.bidirectional), batch_size,  self.hidden_dim,
                                      device = device)
                )
                self.h0 = torch.nn.Parameter(h0, requires_grad = True)

            if self.c0 is None :
                c0 = torch.nn.init.xavier_uniform(
                    torch.zeros(self.n_layer * (1 + self.bidirectional), batch_size,  self.hidden_dim,
                                      device = device)
                )
                self.c0 = torch.nn.Parameter(c0, requires_grad=True)

        if self.init_state == "CLS" :
            if self.h0 is None :
                self.h0 = torch.zeros(self.n_layer * (1 + self.bidirectional), batch_size,  self.hidden_dim,
                                      device = device)

            self.c0 = batch[f'CLS_hidden_states'].unsqueeze(0).repeat(self.n_layer * (1 + self.bidirectional), 1, 1)

        return self.h0, self.c0

    def forward(self, batch : dict) -> dict :

        x = batch[self.input_name]
        batch_size = x.size()[0]

        h0, c0 = self.get_init_states(batch, batch_size)
        output, (h_n, c_n) = self.lstm(x, (h0, c0))

        if self.bidirectional :
            h_n = h_n.view(2, batch_size, self.hidden_dim)
            c_n = c_n.view(2, batch_size, self.hidden_dim)
            #output = output.view(self.batch_size, self.sequence_len, 2, self.hidden_dim)

        batch[f"{self.output_name}_hidden_last"] = h_n
        batch[f"{self.output_name}_cell_last"] = c_n
        batch[f"{self.output_name}"] = output

        return batch

@register("MODULES")
class CRFLayer(CRF, torch.nn.Module) :
    def __init__(self,
                 input_name: str,
                 output_name: str,
                 labels_name: str,
                 hidden_features : int,
                 out_features : int,
                 normalization="none"):

        assert out_features >= 1
        self.out_features = out_features

        assert hidden_features != 1
        self.hidden_features = hidden_features

        super().__init__(out_features)
        super(torch.nn.Module, self).__init__()

        self.crf = CRF(out_features)
        self.normalization = normalization
        self.input_name = input_name
        self.output_name = output_name
        self.labels_name = labels_name
        self.decoder = torch.nn.Linear(self.hidden_features, self.out_features)

    def pad_output(self, outputs,  dimension, device) :
        padded = torch.zeros(dimension, device = device)
        for i, s in enumerate(outputs):
                padded[i][:len(s)] = torch.tensor(s)
        return padded.long()

    def forward(self, model_inputs : dict) -> dict :

        if self.output_name in model_inputs :
            raise RuntimeError(f'Never modify an existing key! ({self.output_name})')

        if self.input_name not in model_inputs:
            raise RuntimeError(f'Key not available. ({self.output_name})')

        features = self.decoder(model_inputs[self.input_name])

        prediction = self.crf.viterbi_tags(features, model_inputs['attention_mask'])

        model_inputs[self.output_name] = features
        model_inputs[self.output_name + "_label"] = self.pad_output(
            [x[0] for x in prediction],
            features.size()[:2],
            features.device
        )

        model_inputs[f"CRF_loss"] = - self.crf(
            features,
            model_inputs[self.labels_name],
            mask = model_inputs['attention_mask']
        )

        return model_inputs


@register("MODULES")
class AnchorEmbeddingsModule(torch.nn.Module) :

    _available_types = [
        "learned",
        "fixed"
    ]

    def __init__(self,
                 anchor_points : int = 10,
                 sharpness : float = 15.0,
                 embedding_type : str = "learned",
                 embedding_size : int = 768,
                 max_sequence_length : int = 512,
                 input_name : str = "word_vector",
                 output_name : str = "embedded_words",
                 **kwargs
                 ):

        super().__init__()

        self.input_name = input_name
        self.output_name = output_name

        self.max_position_embeddings = max_sequence_length

        assert anchor_points > 0, "Anchor point should be positive integer"
        self.anchor_points = anchor_points

        self.sharpness_pos = sharpness

        assert embedding_type in self._available_types, f"{embedding_type} type not surpported."
        self.type = embedding_type

        self.register_buffer("position_indexes", torch.arange(max_sequence_length).expand((1, -1)))


        if self.type == "learned" :
            self.init_learned_anchors(anchor_points, embedding_size)
        elif self.type == "fixed" :
            self.init_anchors(anchor_points, embedding_size)
        else :
            raise RuntimeError(f"custom position embedding type {self.type} not supported")



    def init_learned_anchors(self, anchor_points: int, hidden_size : int):
        """

        """

        # init anchors position to be between 0 and 1
        # anchors_position = torch.empty(config.anchor_points, requires_grad=True) v2
        # anchors_position = torch.arange(0, 1 , 1 / config.anchor_points, requires_grad=True).logit() # v3
        anchors_position = torch.arange(0, 1, 1 / anchor_points, requires_grad=True)  # v4

        # anchors_position.retain_grad()
        # torch.nn.init.uniform_(anchors_position, 0.0, 1.0).logit()  v2
        self.initial_anchor_position = anchors_position.detach().clone()
        self.anchor_position = torch.nn.Parameter(anchors_position)

        # Init Embedding matrix
        anchors_embedding = torch.empty(anchor_points, hidden_size, requires_grad=True)
        torch.nn.init.xavier_uniform_(anchors_embedding)
        self.anchor_embeddings = torch.nn.Parameter(anchors_embedding)

    def init_anchors(self, anchor_points: int, hidden_size : int) :
        """

        """

        # Init Embedding matrix
        anchors_embedding = torch.empty(anchor_points, hidden_size, requires_grad=True)
        torch.nn.init.xavier_uniform_(anchors_embedding)
        self.anchor_embeddings = torch.nn.Parameter(anchors_embedding)

    def anchor_embedding(self, position_ids, batch_size, sequence_lenghts, attention_mask) :

        assert (sequence_lenghts < self.max_position_embeddings).all()

        # get batch representation
        indexes = position_ids.repeat(batch_size, 1) * attention_mask

        # normalize given lenght for relative position
        indexes = indexes / sequence_lenghts

        # for each relative position get the index of the associated anchor embedding
        indexes = (indexes * (self.anchor_points - 1)).round().long()

        # select anchors for each token in the embedding matrix
        return self.anchors_embeddings(indexes)

    def learned_anchor_embedding(
            self,
            batch_size,
            sequence_lenghts,
            attention_mask,
            position_ids = None,
    ) :

        assert (sequence_lenghts < self.max_position_embeddings).all()

        # get batch representation
        indexes = position_ids.repeat(batch_size, 1) * attention_mask

        # normalize given lenght for relative position
        indexes = indexes / sequence_lenghts
        indexes = indexes.unsqueeze(-1).repeat(1, 1, self.anchor_points)

        # processing anchors
        #anchors = self.anchor_position.sigmoid().unsqueeze(0).repeat(batch_size, self.max_position_embeddings, 1)
        anchors = (self.anchor_position / self.anchor_position.max()).unsqueeze(0).repeat(batch_size, self.max_position_embeddings, 1) # v4

        # distance matrix
        dist_matrix = (anchors - indexes).square()

        # attention matrix
        att_matrix = (- dist_matrix * self.sharpness_pos).softmax(dim=-1)

        # retreive anchors embeddings (torch.bmm -> batch mat mul)
        pos_emb = self.anchor_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        return torch.bmm(att_matrix, pos_emb)

    def forward(self, model_inputs: dict) -> dict:

        batch_size = model_inputs[self.input_name].size()[0]
        sequence_lengths = model_inputs['attention_mask'].sum(dim=1).unsqueeze(-1) - 1
        attention_mask = model_inputs['attention_mask']
        position_ids = self.position_indexes


        if self.type == "learned" :
            embeddings =  self.learned_anchor_embedding(
                batch_size,
                sequence_lengths,
                attention_mask,
                position_ids
            )
        elif self.type == "learned" :
            embeddings = self.anchor_embedding(
                batch_size,
                sequence_lengths,
                attention_mask,
                position_ids
            )
        else :
            raise RuntimeError(f"custom position embedding type {self.type} not supported")

        model_inputs[self.output_name] = model_inputs[self.input_name] + embeddings

        return model_inputs




