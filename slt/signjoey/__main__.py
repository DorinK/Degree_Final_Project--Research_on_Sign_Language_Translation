import argparse
import os

import sys
from slt.signjoey.training import train
from slt.signjoey.prediction import test

# sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"#"0,1,2,3"


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument(
        "--output_path", type=str, help="path for saving translation output"
    )
    ap.add_argument("--gpu_id", type=str, default="1", help="gpu to run your job on")
    args = ap.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    # train(cfg_file='/home/nlp/dorink/project/slt/configs/sign.yaml')
    main()


#--output_path ./output_slt train /home/nlp/dorink/project/slt/configs/sign.yaml