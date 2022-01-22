import argparse
import os

import sys
from slt.signjoey.training import train
from slt.signjoey.prediction import test

# TODO: Use all GPU devices.    V
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def main():

    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument("--output_path", type=str, help="path for saving translation output")

    ap.add_argument("--gpu_id", type=str, default="0,1,2,3", help="gpu to run your job on")  # TODO: Use all GPUs devices.  V

    args = ap.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        # TODO: Temporary - I used this code to run the model with the ChicagoFSWild dataset on the test set.   V
        # from training import load_config
        # cfg = load_config(args.config_path)
        # ckpt = "{}/{}.ckpt".format(cfg["training"]["model_dir"], 2600)
        # output_name = "best.IT_{:08d}".format(2600)
        # output_path = os.path.join(cfg["training"]["model_dir"], output_name)
        # test(cfg_file=args.config_path, ckpt=ckpt, output_path=output_path)
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    # train(cfg_file='/home/nlp/dorink/project/slt/configs/sign.yaml')  # TODO: Just checking.  V
    main()
