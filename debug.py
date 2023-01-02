from traindata import gen_train_data
from testdata import gen_test_data
from trainer import My_ClassificationTrainer
from model import Net



if __name__=='__main__':
    buyer_path='./data/buyer.txt'
    edge_path='./data/edge.txt'
    label_path='./data/label.txt'
    train_data = gen_train_data(buyer_path,edge_path,label_path)
    test_data = gen_test_data(buyer_path,edge_path,label_path)
    m = Net().cpu()
    a = My_ClassificationTrainer()
    a.model = m
    a.train(train_data[0],'cpu',None)
    m.eval()
    m(test_data)
