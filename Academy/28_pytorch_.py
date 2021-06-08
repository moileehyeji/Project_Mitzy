# https://medium.com/@inmoonlight/pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%95%98%EA%B8%B0-cnn-62a9326111ae
# https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/
import db_connect as db
import numpy as np
import pandas as pd
import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset

def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity','person', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    # pred = df.iloc[:,1:-1]

    if is_train == True:
        # train, test 나누기
        train_value = df[ '2020-09-01' > df['date'] ]
        x = train_value.iloc[:,1:-1].astype('float64')
        y = train_value.iloc[:,-1].astype('float64').to_numpy()
    else:
        test_value = df[df['date'] >=  '2020-09-01']
        x = test_value.iloc[:,1:-1].astype('float64')
        y = test_value.iloc[:,-1].astype('float64').to_numpy()

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x = pd.get_dummies(x, columns=["category", "dong"]).to_numpy()
    # 카테고리랑 동만 원핫으로 해준다 

    return x, y

# torch의 Dataset 을 상속.
class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return {"x":torch.tensor(self.x_data[index], dtype=torch.float32), 
                "y":torch.tensor(self.y_data[index], dtype=torch.float32)}
        # return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len

x_pred, y_pred = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date ", is_train = False)
x_train, y_train = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC")

# 학습 데이터, 시험 데이터 배치 형태로 구축하기
trainsets = TensorData(x_train, y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

testsets = TensorData(x_pred, y_pred)
testloader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=False)

# for data in trainloader:
#     print(data['x'].shape)  #torch.Size([32, 47])

# construct model  on cuda if available
use_cuda = torch.cuda.is_available()

class MyModel(nn.Module):
    
    def __init__(self, X_dim, y_dim):
        super(MyModel, self).__init__()
        layer1 = nn.Linear(X_dim, 128)
        activation1 = nn.ReLU()
        layer2 = nn.Linear(128, y_dim)
        self.module = nn.Sequential(
            layer1,
            activation1,
            layer2
        )
        
    def forward(self, x):
        out = self.module(x)
        result = F.softmax(out, dim=1)
        return result    


# 준비재료
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-5
optimizer = optim.SGD(model.paraeters(), lr=learning_rate)
num_epochs = 2
num_batches = len(train_loader)

for epoch in range(num_epochs):
	for i, data in enumerate(train_loader):
		x, x_labels = data # x.size() = [batch, channel, x, y]
		# init grad
		optimizer.zero_grad() # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨
		# forward
		pred = model(x)
		# calculate loss
		loss = criterion(pred, x_labels)
		# backpropagation
		loss.backward()
		# weight update
		optimizer.step()
		# 학습과정 출력
		running_loss += loss.item()
		if (i+1)%2000 == 0: # print every 2000 mini-batches
			print("epoch: {}/{} | step: {}/{} | loss: {:.4f}".format(epoch, num_epochs, i+1, num_batches, running_loss/2000))
			running_loss = 0.0

print("finish Training!")    

""" class CNNClassifier(nn.Module):
    
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv1d(32, 32, 2, 2) #in_channels , out_channels , kernel_size , stride = 1
        
        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU()
        )
        
        # fc1 = nn.Linear(32*4*4, 32)
        # activation ReLU
        fc2 = nn.Linear(32, 16)
        # activation ReLU
        fc3 = nn.Linear(16, 8)
        fc4 = nn.Linear(8, 1)

        self.fc_module = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3,
            nn.ReLU(),
            fc4
        )
        
        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda() 
 """

'''
class MapDataset(Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        return {"x":torch.tensor([idx], 
                                     dtype=torch.float32), 
                "y": torch.tensor(idx, 
                                      dtype=torch.float32)}

train_value = load_data("SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC")
test_value = load_data("select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC",is_train=False)

train_dataset = MapDataset(train_value)
test_dataset = MapDataset(test_value)


batch_size = 500
train_loader = torch.utils.data.DataLoader(train_value,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_value,batch_size=batch_size, shuffle=True)

print(train_loader)
# print(test_loader)

'''