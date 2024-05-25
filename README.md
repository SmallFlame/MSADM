## Requirements
Use python 3.11 from MiniConda
- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0
## data
data: We obtained the dataset through ns3 simulation url:
## run
modelutil/**model.py Corresponding to different types of timing prediction models
00normal.py:
01initdata.py
02Train.py
03gentext.py
04nortest.py

