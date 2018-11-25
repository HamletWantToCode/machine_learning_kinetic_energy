#!/bin/bash

python=/home/hongbin/anaconda3/envs/project-qml/bin/python
date > out

cd tools/
(time nohup $python database.py &) >>../out 2>&1

