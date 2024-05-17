# healthmanagement

nohup /root/anaconda3/envs/py37_ns/bin/python /root/public/wxn/healthmanagement/model/02RunTransformer.py 02RunTransformer.log 2>&1 &   3240361

nohup /root/anaconda3/envs/py37_ns/bin/python model/03RunLstm.py > 03RunLstm.log 2>&1 &  2986936

nohup /root/anaconda3/envs/py37_ns/bin/python model/04RunAttentionLstm.py > 04RunAttentionLstm.log 2>&1 & 2987129

nohup /root/anaconda3/envs/py37_ns/bin/python model/05lstmTransformer.py > 05lstmTransformer.log 2>&1 & 2987238

nohup /root/anaconda3/envs/py37_ns/bin/python model/06e2elstm.py > 06e2elstm.log 2>&1 & 3240658

nohup /root/anaconda3/envs/py37_ns/bin/python /root/public/wxn/healthmanagement/model/07CNN.py > 07CNN.log 2>&1 &  3778628 3779199

nohup /root/anaconda3/envs/py37_ns/bin/python /root/public/wxn/healthmanagement/model/02Train.py > 02Train.log 2>&1 & 920971


nohup /root/anaconda3/envs/py37_ns/bin/python model/03printRoc.py > 03printRoc.log 2>&1 &
55661