#!/bin/bash

python=/anaconda3/envs/workspace/bin/python
date > out

(time nohup $python tools/database.py &) >>out 2>&1

