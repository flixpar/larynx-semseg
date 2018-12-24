import yaml
import argparse
import numpy as np

import torch
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True


def validate(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader, batch_size=cfg["training"]["batch_size"], num_workers=8)
    running_metrics = runningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)    
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(valloader):

        images = images.to(device)
        gt = labels.numpy()

        outputs = model(images).data.cpu().numpy()

        flipped_images = torch.flip(images, dims=(3,))
        outputs_flipped = model(flipped_images)
        outputs_flipped = torch.flip(outputs_flipped, dims=(3,)).data.cpu().numpy()

        outputs = (outputs + outputs_flipped) / 2.0
        pred = np.argmax(outputs, axis=1)

        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/unet_larynx.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="unet_larynx.pkl",
        help="Path to the saved model",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
