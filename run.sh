#!/bin/bash

python=/home/hongbin/anaconda3/envs/project-qml/bin/python
date > out

(time nohup $python tools/database.py &) >>out 2>&1

