#! /bin/bash

pip3 install -r /home/nlp/dorink/project/FS-Detection/requirements.txt --no-cache-dir
pip3 install warpctc-pytorch==0.2.2+torch11.cpu -f https://github.com/espnet/warp-ctc/releases/tag/v0.2.2 --no-cache_dir
