#!/usr/bin/env bash

curl https://bootstrap.pypa.io/get-pip.py | python3.9
pip install -r requirements.txt

python3.9 model.py

# have model.py write to output files, then you can use
#    curl bashupload.com -T your_file.txt
# to upload
