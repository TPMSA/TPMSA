import torch
from ignite.metrics import Metric
from ignite.metrics import Accuracy, MeanAbsoluteError
from sklearn.metrics import f1_score


def binary_transform(output):
    y_pred, y = output
    non_zeros = torch.nonzero(y, as_tuple=True)
    binary_y = torch.gt(y[non_zeros], 0).int()
    binary_preds = torch.gt(y_pred[non_zeros], 0).int()
    return binary_preds, binary_y


def five_transform(output):
    y_pred, y = output
    y = torch.round(torch.clamp(y, -2, 2))
    y_pred = torch.round(torch.clamp(y_pred, -2, 2))
    binary_y = (y == y_pred).int()
    binary_pred = torch.ones_like(binary_y)
    return binary_pred, binary_y


def seven_transform(output):
    y_pred, y = output
    y = torch.round(torch.clamp(y, -3, 3))
    y_pred = torch.round(torch.clamp(y_pred, -3, 3))
    binary_y = (y == y_pred).int()
    binary_pred = torch.ones_like(binary_y)
    return binary_pred, binary_y


class Pearson(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._y = None
        self._pred = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._y = list()
        self._pred = list()
        super().reset()

    def update(self, output):
        y_pred, y = output
        self._y.append(y)
        self._pred.append(y_pred)

    def compute(self):
        y_pred = torch.cat(self._pred, 0)
        y = torch.cat(self._y, 0)
        v_pred = y_pred - torch.mean(y_pred)
        v_y = y - torch.mean(y)
        corr = torch.sum(v_pred * v_y) / ((torch.sqrt(torch.sum(v_pred ** 2))) * (torch.sqrt(torch.sum(v_y ** 2))))
        return corr


class F1(Metric):
    def __init__(self, output_transform=binary_transform):
        self._y = None
        self._pred = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._y = list()
        self._pred = list()
        super().reset()

    def update(self, output):
        y_pred, y = output
        self._y.append(y)
        self._pred.append(y_pred)

    def compute(self):
        y_pred = torch.cat(self._pred, 0).cpu()
        y = torch.cat(self._y, 0).cpu()
        score = f1_score(y, y_pred, average='weighted')
        return score


val_metrics = {'acc2': Accuracy(output_transform=binary_transform),
               'acc5': Accuracy(output_transform=five_transform),
               'acc7': Accuracy(output_transform=seven_transform),
               'f1': F1(),
               'corr': Pearson(),
               'mae': MeanAbsoluteError()}
