import logging
import torch
import torch.nn as nn
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from transformers import AutoProcessor, CLIPModel

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip')
class CLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        # base - 768, large 1024
        self.head = nn.Linear(1024, 2)
        
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
    def build_backbone(self, config):
        # prepare the backbone
        # clip-vit-large-patch14
        # clip-vit-base-patch16
        _, backbone = get_clip_visual(model_name="openai/clip-vit-large-patch14")
        return backbone

        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    # @torch.no_grad()
    def features(self, data_dict: dict) -> torch.tensor:
        with torch.no_grad():
            feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        with torch.no_grad():
            features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)
        return pred_dict


def get_clip_visual(model_name = "openai/clip-vit-large-patch14"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model
