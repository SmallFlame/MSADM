## Requirements
python = 3.70
conda env create -n msadm -f environment.yml
conda activate msadm
## data
data: We obtained the dataset through ns3 simulation url:
## run
modelutil/**model.py Corresponding to different types of timing prediction models
00normal.py:
01initdata.py
02Train.py
03gentext.py
04nortest.py

