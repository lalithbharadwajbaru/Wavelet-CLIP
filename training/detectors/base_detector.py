import abc
import torch.nn as nn

class AbstractDetector(nn.Module, metaclass=abc.ABCMeta):
    """
    All deepfake detectors should subclass this class.
    """
    def __init__(self, config=None, load_param=False):
        """
        config:   (dict)
            configurations for the model
        load_param:  (False | True | Path(str))
            False Do not read; True Read the default path; Path Read the required path
        """
        super().__init__()

    @abc.abstractmethod
    def features(self, data_dict):
        """
        Returns the features from the backbone given the input data.
        """
        pass

    @abc.abstractmethod
    def forward(self, data_dict, inference=False):
        """
        Forward pass through the model, returning the prediction dictionary.
        """
        pass

    @abc.abstractmethod
    def classifier(self, features):
        """
        Classifies the features into classes.
        """
        pass

    @abc.abstractmethod
    def build_backbone(self, config):
        """
        Builds the backbone of the model.
        """
        pass

    @abc.abstractmethod
    def build_loss(self, config):
        """
        Builds the loss function for the model.
        """
        pass

    @abc.abstractmethod
    def get_losses(self, data_dict, pred_dict):
        """
        Returns the losses for the model.
        """
        pass

    @abc.abstractmethod
    def get_train_metrics(self, data_dict, pred_dict):
        """
        Returns the training metrics for the model.
        """
        pass
