#!/bin/bash

# python=/storage1/home/renhb/anaconda2/envs/py36/bin/python
python=/Users/hongbinren/anaconda2/envs/py36/bin/python
export TOOLS=/Users/hongbinren/Documents/program/1D/tools

date > out

# echo 'Construct database...' >> out
# (time $python database.py) >> out 2>&1
# echo 'Database finished...' >> out

echo 'Model optimize...' >> out
(time $python $TOOLS/optimization.py > result) >>out 2>&1
echo 'Finish optimize...' >> out

date >> out
