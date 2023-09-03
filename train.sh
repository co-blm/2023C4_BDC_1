#! /bin/bash  # employ bash shell
python ./code/generate_label.py
python ./code/feature_extract.py
python ./code/train.py