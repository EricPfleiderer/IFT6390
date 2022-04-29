from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
import torch


class DataTransform(ABC):

    @abstractmethod
    def __init__(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, x, y):
        pass

    @staticmethod
    def validate_shape(x):
        # Batch the single sequence as a column vector
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=1)
        # Inverse sequences from row to column vectors
        elif len(x.shape) == 2:
            x = torch.permute(x, (1, 0))

        elif len(x.shape) == 3:
            x = torch.squeeze(x, dim=-1)
            x = torch.permute(x, (1, 0))
        else:
            raise ValueError('Parameter shape invalid.')
        return x


class FitTransform(DataTransform):

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
        self.scaler = MinMaxScaler((-1, 1))
        self.fit(Scaler.validate_shape(x))

    def transform(self, x):
        x = Scaler.validate_shape(x)
        z = torch.tensor(self.scaler.transform(x), dtype=torch.float)
        return torch.squeeze(z, dim=1)

    def inverse_transform(self, x, y):
        x = Scaler.validate_shape(y)
        z = torch.tensor(self.scaler.inverse_transform(x))
        return torch.squeeze(z, dim=1)

    def fit(self, x):
        return self.scaler.fit(x)


class LogTransform(DataTransform):

    def __init__(self, x):
        pass

    # TODO: consider clipping for numerical stability
    def transform(self, x):
        # TODO: instead of x+1, change 0 values for avg with surrounding values
        return torch.squeeze(torch.log(x+1), dim=0)

    def inverse_transform(self, x, y):
        return torch.squeeze(torch.exp(y) - 1, dim=0)


class LDiff(DataTransform):

    def __init__(self, x):
        self.lag = 24
        self.first_elements = x[:self.lag]

    # Differences a times series by lag
    def difference(self, x, i):
        return x[i-self.lag] - x[i]

    # Used to invert a differenced sequence, given the beginning of the undifferenced sequence
    def addition(self, z, i):
        return z[i-self.lag] + z[i]

    def transform(self, x):

        """
        Converts dimensionality of sequences from len(s) to len(s)-24 (lagg)

        :param x:
        :return:
        """

        new_x = torch.empty((0,))
        for i in range(self.lag, x.shape[0]):
            new_x = torch.cat((new_x, torch.unsqueeze(self.difference(x, i), dim=0)))
        return new_x

    def inverse_transform(self, x, y):
        new_x = torch.squeeze(x, dim=0)
        new_x = torch.squeeze(new_x, dim=1)
        full_seq = torch.cat((new_x, y))
        new_z = self.addition(full_seq, i=len(full_seq)-1)
        return new_z


def apply_transforms(x: torch.Tensor, transforms: list) -> (torch.Tensor, list):

    """
    Applies transforms on a batch of sequences.
    :param x: 2D tensor representing a batch of unprocessed sequences.
    :param transforms: List of Transform callables.
    :return: Transformed x, 2D list of fitted transformers
    """
    new_x = [torch.squeeze(x.clone(), dim=0)]
    transformers = []

    for i in range(len(new_x)):
        tfs = []
        for transform in transforms:
            transformer = transform(new_x[i])  # Initializing object from transform class
            new_x[i] = transformer.transform(new_x[i])
            tfs.append(transformer)
        transformers.append(tfs)
    return torch.unsqueeze(new_x[0], dim=0), transformers


def apply_inverse_transforms(x: torch.Tensor, y: torch.Tensor, transformers) -> torch.Tensor:

    """
    Reverses the transforms applied to the batch x.
    :param x: Transformed 2D tensor representing a batch of transformed sequences.
    :param y: Model prediction to be converted back to original space.
    :param transformers: 2D list with rows of fitted transformers (in the order they were applied).
    :return: Untransformed sequences
    """

    new_y = y.clone()
    reversed = transformers.copy()
    for tfs in reversed:
        tfs.reverse()

    for i in range(len(new_y)):
        for tfs in reversed:
            for tf in tfs:
                new_y[i] = tf.inverse_transform(x, new_y[i])

    return new_y


