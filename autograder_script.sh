#!/usr/bin/env bash

# pip install -r requirements.txt

source /autograder/source/env/bin/activate

python3.7 model.py

# have model.py write to output files, then you can use
#    curl bashupload.com -T your_file.txt
# to upload
