#!/usr/bin/env bash

curl https://pypi.org

python3 model.py

# have model.py write to output files, then you can use
#    curl bashupload.com -T your_file.txt
# to upload
