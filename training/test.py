import numpy as np
import random
import yaml
import logging
import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm
from metrics.utils import get_test_metrics
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_seed(config):
    """Initialize random seed for reproducibility."""
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

def prepare_testing_data(config):
    """Prepare DataLoader for testing datasets."""
    def get_test_data_loader(test_name):
        config_copy = config.copy()
        config_copy['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=config_copy, mode='test')
        return torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False
        )

    test_data_loaders = {}
    for test_name in config['test_dataset']:
        test_data_loaders[test_name] = get_test_data_loader(test_name)
    return test_data_loaders

def choose_metric(config):
    """Choose the metric for evaluation."""
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError(f'Metric {metric_scoring} is not implemented.')
    return metric_scoring

def test_one_dataset(model, data_loader):
    """Test the model on a single dataset and return predictions and metrics."""
    cls_prediction_lists, prediction_lists, feature_lists, label_lists = [], [], [], []
    
    for data_dict in tqdm(data_loader, total=len(data_loader)):
        data, label, mask, landmark = (
            data_dict['image'], 
            data_dict['label'], 
            data_dict['mask'], 
            data_dict['landmark']
        )
        label = torch.where(data_dict['label'] != 0, 1, 0)

        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        predictions = inference(model, data_dict)
        label_lists.extend(label.cpu().numpy())
        prediction_lists.extend(predictions['prob'].cpu().numpy())
        cls_prediction_lists.extend(predictions['cls'].cpu().numpy())
        feature_lists.extend(predictions['feat'].cpu().numpy())

    return np.array(prediction_lists), np.array(cls_prediction_lists), np.array(label_lists), np.array(feature_lists)

def test_epoch(model, test_data_loaders):
    """Run a test epoch across all datasets."""
    model.eval()
    metrics_all_datasets = {}

    for dataset_name, data_loader in test_data_loaders.items():
        data_dict = data_loader.dataset.data_dict
        predictions_nps, cls_pred_nps, label_nps, feat_nps = test_one_dataset(model, data_loader)

        metric_one_dataset = get_test_metrics(
            y_pred=predictions_nps, y_true=label_nps, img_names=data_dict["image"]
        )
        metrics_all_datasets[dataset_name] = metric_one_dataset

        logger.info(f"Dataset: {dataset_name}")
        for metric, value in metric_one_dataset.items():
            logger.info(f"{metric}: {value}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    """Perform inference with the model."""
    return model(data_dict, inference=True)

def load_model(config, weights_path):
    """Load the model with the specified weights."""
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    
    if weights_path:
        try:
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = {k[len('module.'):]: v for k, v in ckpt.items()}
            model.load_state_dict(state_dict, strict=True)
            logger.info('Checkpoint loaded successfully!')
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            exit(1)
    else:
        logger.warning('No pre-trained weights provided.')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Evaluate a pre-trained model.')
    parser.add_argument("--detector_path", type=str, default="./training/config/detector/clip.yaml", help="Path to detector YAML file.")
    parser.add_argument("--test_dataset", nargs="+", help="List of test datasets.")
    parser.add_argument("--weights_path", type=str, default="./training/weights", help="Path to model weights.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configurations
    try:
        with open(args.detector_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {args.detector_path} not found.")
        exit(1)

    try:
        with open('./training/config/test_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Test configuration file not found.")
        exit(1)

    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']

    # Update config with command-line arguments
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
        config2['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        config2['weights_path'] = args.weights_path

    # Initialize seed
    init_seed(config)

    # Set cuDNN benchmark
    if config.get('cudnn', False):
        cudnn.benchmark = True

    # Prepare testing data loaders
    new_config = {**config, **config2}
    test_data_loaders = prepare_testing_data(new_config)

    # Load the model
    model = load_model(config, args.weights_path)

    # Start testing
    best_metric = test_epoch(model, test_data_loaders)
    logger.info('Testing completed!')

if __name__ == '__main__':
    main()
