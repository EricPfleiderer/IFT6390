from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
import torch


class DataTransform(ABC):

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, z):
        pass


class FitTransform(DataTransform):

    @abstractmethod
    def __init__(self, x):
        pass

    @abstractmethod
    def fit(self, x):
        pass


class Scaler(FitTransform):

    def __init__(self, x):
        """
        expects 2d data, so must batch x if it is one dim
        normalizes dim 0, so sequences must be columns
        :param x: Single sequence, or 2d array with row sequences.
        """

        self.scaler = MinMaxScaler()
        self.fit(Scaler.validate_shape(x))

    @staticmethod
    def validate_shape(x):

        # Batch the single sequence as a column vector
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=1)

        # Inverse sequences from row to column vectors
        elif len(x.shape) == 2:
            x = torch.permute(x, (1, 0))
        return x

    def transform(self, x):
        x = Scaler.validate_shape(x)
        z = torch.tensor(self.scaler.transform(x), dtype=torch.float)
        return torch.squeeze(z, dim=1)

    def inverse_transform(self, z):
        if len(z.shape) == 1:
            z = torch.unsqueeze(z, dim=1)
        return torch.tensor(self.scaler.inverse_transform(z))

    def fit(self, x):
        return self.scaler.fit(x)


class LogTransform(DataTransform):
    # TODO: consider clipping for numerical stability
    def transform(self, x):
        # TODO: instead of x+1, change 0 values for avg with surrounding values
        return torch.log(x+1)

    def inverse_transform(self, z):
        return torch.exp(z) - 1


class LDiff(DataTransform):
    pass


def apply_transforms(x: torch.Tensor, transforms: list) -> (torch.Tensor, list):

    """
    Applies transforms on a batch of sequences.
    :param x: 2D tensor representing a batch of unprocessed sequences.
    :param transforms: List of Transform callables.
    :return: Transformed x, 2D list of fitted transformers
    """
    new_x = x.clone()
    transformers = []

    for i in range(len(new_x)):
        tfs = []
        for transform in transforms:
            if issubclass(transform, FitTransform):
                transformer = transform(new_x[i])
            else:
                transformer = transform()
            new_x[i] = transformer.transform(new_x[i])
            tfs.append(transformer)
        transformers.append(tfs)
    return new_x, transformers


def apply_inverse_transforms(x: torch.Tensor, transformers) -> torch.Tensor:

    """
    Reverses the transforms applied to the batch x.
    :param x: Transformed 2D tensor representing a batch of transformed sequences.
    :param transformers: 2D list with rows of fitted transformers (in the order they were applied).
    :return: Untransformed sequences
    """

    new_x = x.clone()
    reversed = transformers.copy()
    for tfs in reversed:
        tfs.reverse()

    for i in range(len(new_x)):
        for tfs in reversed:
            for tf in tfs:
                new_x[i] = tf.inverse_transform(new_x[i])

    return new_x


