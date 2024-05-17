from modelutil.MSADM import train,test
# from modelutil.LstmModel import train,test
# from modelutil.AttentionLstm import train,test
# from modelutil.LSTMTransformerModel import train,test
# from modelutil.CNNModel import train,test

from config import trainConfig
tra_math_path = trainConfig["tra_math_path"]
tra_ruler_path = trainConfig["tra_ruler_path"]
val_math_path = trainConfig["val_math_path"]
val_ruler_path = trainConfig["val_ruler_path"]
learning_rate = trainConfig["learning_rate"]
batch_size = trainConfig["batch_size"]
epoch_num = trainConfig["epoch_num"]
def Train():
    model_path = 'out/model/TransformerRE'
    train(tra_math_path,tra_ruler_path,learning_rate,batch_size,40,model_path,init=True)
def Test():
    for i in range(60):
        result_path = f'out/test/TransformerRE/res{i+1}.csv'
        model_path = f'out/model/TransformerRE/model{i+1}.pkl'
        test(val_math_path,val_ruler_path,result_path,model_path)
Train()
Test()