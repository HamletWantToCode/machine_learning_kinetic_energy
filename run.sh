#!/bin/bash

python=/anaconda3/envs/workspace/bin/python
date > out

cd tools/
(time nohup $python database.py &) >>../out 2>&1

