import os
import numpy as np
import cv2
import random
import yaml
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from detectors import DETECTOR
from torch.utils.data import Dataset
from torchvision import transforms as T

import cv2
from PIL import Image
from sklearn import metrics

import argparse

parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument(
    "--detector_path",
    type=str,
    default="./training/config/detector/clip.yaml",
    help="path to detector YAML file",
)
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--weights_path", type=str, default="./training/weights")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config["manualSeed"] is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config["cuda"]:
        torch.cuda.manual_seed_all(config["manualSeed"])


def get_test_metrics(y_pred, y_true):
    y_pred = y_pred.squeeze()
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    return {
        "acc": acc,
        "auc": auc,
        "eer": eer,
        "ap": ap,
        "pred": y_pred,
        "label": y_true,
    }


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, config, transform=None):
        self.transform = transform
        ####### Generated Image ##########
        self.img_dir = img_dir
        self.img_labels = np.ones(len(os.listdir(self.img_dir)))
        self.img_files = sorted(os.listdir(self.img_dir))
        ### Real Images #####
        self.celab_dir = "./sampled_celebahq_50K/CelebA_Real/"
        self.celaba_real = sorted(os.listdir(self.celab_dir))
        self.real_labels = np.zeros(len(os.listdir(self.celab_dir)))
        ######## Merge both Generated and Real Images #########
        self.image_list = self.img_files + self.celaba_real
        self.label_list = np.array(list(self.img_labels) + list(self.real_labels))
        self.size = config["resolution"]
        self.config = config

        assert (
            len(self.img_files) != 0 and len(self.img_labels) != 0
        ), f"Collect nothing for test mode!"

    def __len__(self):
        print(f"Total Testing Images for {len(self.image_list)}", flush=True)
        return len(self.image_list)

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config["mean"]
        std = self.config["std"]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def __getitem__(self, idx):
        label = self.label_list[idx]

        if label == 0:
            img_path = os.path.join(self.celab_dir, self.image_list[idx])
        else:
            img_path = os.path.join(self.img_dir, self.image_list[idx])

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Loaded image is None: {}".format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        pil_img = Image.fromarray(np.array(img, dtype=np.uint8))
        image_trans = self.normalize(self.to_tensor(pil_img))
        return image_trans, torch.tensor(label, dtype=torch.long)


# face_dataset = CustomImageDataset(img_dir ="/home/rohit/Diffusion/LDM_out")
def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        gen_data = {
            "LDM": "./sampled_celebahq_50K/LDM",
            "DDPM": "./sampled_celebahq_50K/DDPM",
            "DDIM": "./sampled_celebahq_50K/DDIM",
        }
        # update the config dictionary with the specific testing dataset
        config = (
            config.copy()
        )  # create a copy of config to avoid altering the original one
        config["test_dataset"] = test_name  # specify the current test dataset
        test_set = CustomImageDataset(config=config, img_dir=gen_data[test_name])
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config["test_batchSize"],
            shuffle=False,
            num_workers=int(config["workers"]),
            drop_last=False,
        )
        return test_data_loader

    test_data_loaders = {}
    test_dataset = ["DDIM", "DDPM", "LDM"]
    for one_test_name in test_dataset:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config["metric_scoring"]
    if metric_scoring not in ["eer", "auc", "acc", "ap"]:
        raise NotImplementedError("metric {} is not implemented".format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    cls_prediction_lists = []
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, (img, lab) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data_dict = {}
        # get data
        data = img
        label = lab
        label = torch.where(label != 0, 1, 0)

        # move data to GPU
        data_dict["image"] = data.to(device)
        data_dict["label"] = label.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict["label"].cpu().detach().numpy())
        prediction_lists += list(predictions["prob"].cpu().detach().numpy())
        cls_prediction_lists += list(predictions["cls"].cpu().detach().numpy())
        feature_lists += list(predictions["feat"].cpu().detach().numpy())

    return (
        np.array(prediction_lists),
        np.array(cls_prediction_lists),
        np.array(label_lists),
        np.array(feature_lists),
    )


def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        # compute loss for each dataset
        predictions_nps, cls_pred_nps, label_nps, feat_nps = test_one_dataset(
            model, test_data_loaders[key]
        )
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps)
        metrics_all_datasets[key] = metric_one_dataset

        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
    with open("./training/config/test_config.yaml", "r") as f:
        config2 = yaml.safe_load(f)
    if "label_dict" in config:
        config2["label_dict"] = config["label_dict"]
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
        config2["test_dataset"] = args.test_dataset
    if args.weights_path:
        config["weights_path"] = args.weights_path
        config2["weights_path"] = args.weights_path
        weights_path = args.weights_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config["cudnn"]:
        cudnn.benchmark = True
    new_config = {**config, **config2}
    test_data_loaders = prepare_testing_data(new_config)

    # prepare the model (detector)
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split("/")[-1].split(".")[0].split("_")[2])
        except:
            epoch = 0

        ckpt = torch.load(weights_path, map_location=device)
        remove_prefix = "module."
        state_dict = {
            k[len(remove_prefix) :] if k.startswith(remove_prefix) else k: v
            for k, v in ckpt.items()
        }
        model.load_state_dict(state_dict, strict=True)
        print("===> Load checkpoint done!")
    else:
        print("Fail to load the pre-trained weights")

    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print("===> Test Done!")


if __name__ == "__main__":
    main()
