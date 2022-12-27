import torch
from source.losses.base_loss import BaseLoss
from source.utils.register import register
import math
from argparse import ArgumentParser
from typing import List

@register('LOSSES')
@register('METRICS')
class BCELoss(BaseLoss, torch.nn.BCELoss):
    _names = ['BinaryCrossEntropy', 'BCE']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        torch.nn.BCELoss.__init__(self, reduction='none')

    def __call__(self, model_output, loss_input):

        #print(model_output)
        loss = super().__call__(*self._prep_inputs(model_output, loss_input))

        #print(f"w : {loss_input['weights'].size()}")
        #print(f"l : {loss.size()}")

        if 'weights' in loss_input :
            loss *= loss_input['weights']

        if 'labels_mask' in loss_input and loss_input['labels_mask'].shape == loss.shape :
            loss = loss.masked_select(loss_input['labels_mask'].bool())

        return loss.mean()

    @staticmethod
    def _prep_inputs(model_outputs, loss_input):

        prd, tgt = model_outputs['prediction'].squeeze(-1).float(), loss_input['labels'].float()

        return prd, tgt

    @staticmethod
    def add_loss_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Accuracy metric')

        group.add_argument("--BCE_reduction", type=str, default='mean', choices = ['none', 'mean', 'sum'])
        group.add_argument("--BCE_weights", type=list, nargs="+", default=[])
        return parser

@register('LOSSES')
class CRFLoss(BaseLoss):
    _names = ['CRF']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, model_output, loss_input) :
        return model_output["CRF_loss"]

@register('METRICS')
class Accuracy(BaseLoss, torch.nn.Module):
    _names = ['Accuracy', 'Acc']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = 0.5

    def __call__(self, model_output, loss_input) :
        prd, tgt = self._prep_inputs(model_output, loss_input)

        mask = model_output['labels_mask'].bool()

        # not selecting backgroud label 0
        prd = prd.masked_select(mask)
        tgt = tgt.masked_select(mask)

        return (((prd > self.threshold) == tgt).sum() / tgt.size()[0]).item()

    @staticmethod
    def _prep_inputs(model_outputs, loss_input):
        prd, tgt = model_outputs['prediction'].squeeze(-1).float(), loss_input['labels'].float()

        return prd, tgt

@register('METRICS')
class AccuracyMultiClass(BaseLoss, torch.nn.Module):
    _names = ['AccuracyMultiClass', 'AccMC']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = 0.5

    def __call__(self, model_output, loss_input) :
        prd, tgt =  model_output['prediction_label'], loss_input['labels']

        # processing mask to ignore background label 0
        mask_prd = prd != 0
        mask_tgt = tgt != 0
        mask = torch.logical_or(mask_tgt, mask_prd)

        if 'attention_mask' in loss_input and (loss_input['attention_mask'].shape == prd.shape):
            mask = torch.logical_and(mask, loss_input['attention_mask'].bool())

        # not selecting backgroud label 0
        prd = prd.masked_select(mask)
        tgt = tgt.masked_select(mask)

        return (prd  == tgt).float().mean()

@register('METRICS')
class Dice(BaseLoss, torch.nn.Module):
    _names = ['DiceScore']

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = 0.5

    def __call__(self, model_output, loss_input) :
        prd, tgt =  model_output['prediction_label'], loss_input['labels']

        prd = prd.view(-1)
        tgt = tgt.view(-1)

        intersection = (prd * tgt).sum().float()
        # TODO : implement multiclass dice loss
        return None

@register('LOSSES')
@register('METRICS')
class MAELoss(BaseLoss, torch.nn.L1Loss):
    _names = ['MAE', 'L1', 'L1Loss']

    def __init__(self, log_name : str = None):
        super().__init__(log_name)

    def __call__(self, model_output, loss_input):
        return super().__call__(*self._prep_inputs(model_output, loss_input))


@register('LOSSES')
@register('METRICS')
class MSELoss(BaseLoss, torch.nn.MSELoss):
    _names = ['MSE', 'L2']

    def __init__(self, log_name : str = None):
        super().__init__(log_name)

    @property
    def abbrv(self):
        return 'L2'

    def __call__(self, model_output, loss_input):
        return super().__call__(*self._prep_inputs(model_output, loss_input))


@register('LOSSES')
@register('METRICS')
class RMSELoss(MSELoss):
    """
    See:
        https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    """

    _names = ['RMSE']

    def __init__(self, log_name : str = None):
        super().__init__(log_name)

    @property
    def abbrv(self):
        return 'RMSE'

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.eps = 1e-6

    def __call__(self, model_output, loss_input):
        _mse = super().__call__(model_output, loss_input)
        return torch.sqrt(_mse + self.eps)


@register('LOSSES')
@register('METRICS')
class CrossEntropyLoss(BaseLoss, torch.nn.CrossEntropyLoss):
    _names = ['CrossEntropy']


    def __init__(self, log_name : str = None, **kwargs):
        super().__init__(log_name)
        torch.nn.CrossEntropyLoss.__init__(self, reduction='none')

    @property
    def name(self):
        if self.log_name is not None:
            return f"{__class__.__name__}_{self.log_name}"
        else:
            return __class__.__name__

    @property
    def abbrv(self):
        return 'xent'

    def __call__(self, model_output, loss_input):

        prd, tgt = self._prep_inputs(model_output, loss_input)

        #print(f"prd shape : {prd.shape}")
        #print(f"tgt shape : {tgt.shape}")
        #print(f"mask shape : {prd.shape}")

        loss = super().__call__(prd, tgt, )

        if 'weights' in loss_input :
            loss *= loss_input['weights']

        sample_loss = loss

        if ('labels_mask' in loss_input) and (loss_input['labels_mask'].shape == loss.shape):
            loss = loss.masked_select(loss_input['labels_mask'].bool())

        return loss.mean(), sample_loss

    @staticmethod
    def _prep_inputs(model_outputs, loss_input):
        prd, tgt = model_outputs['prediction'], loss_input['labels']

        return prd, tgt


@register('LOSSES')
@register('METRICS')
class LogCoshLoss(BaseLoss, torch.nn.Module):
    """
    This regression loss comes from the following repo:
            https://github.com/tuantle/regression-losses-pytorch
    """

    _names = ['LogCosh']

    def __init__(self, log_name : str = None):
        super().__init__(log_name)


    @property
    def abbrv(self):
        return 'LgCsh'

    def __init__(self, reduction='mean', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert reduction in {'mean', 'none'}
        self.reduction = reduction
        self.eps = 1e-12

    @staticmethod
    def _compute_log_cosh(x):
        """
        Numerically stable version of logcosh
        """
        log2 = torch.full(x.shape, math.log(2), device=x.device)
        return torch.logaddexp(x, -x) - log2

    def forward(self, model_output, loss_input):
        src, tgt = self._prep_inputs(model_output, loss_input)
        logcosh = self._compute_log_cosh(src - tgt + self.eps)
        if self.reduction == 'mean':
            logcosh = torch.mean(logcosh)
        return logcosh
