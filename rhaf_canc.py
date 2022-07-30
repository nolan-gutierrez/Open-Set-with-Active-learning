import sys
import os
from sklearn import metrics
#importing the libraries
import torch

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pdb import set_trace as trace

#importing the dataset
import random

seed = 123
"""
https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43
"""

data_file_path = './wisconsin_breast_cancer.csv'
def load_data(test_split=0.20):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(data_file_path):
        download_data_file()

    assert os.path.exists(data_file_path), \
        "%s - unable to open file!" % data_file_path

    wis_df = pd.read_csv(data_file_path, index_col=0)
    print(f"wis_df.shape: {wis_df.shape}")

    # diagnosis is the target col - char
    wis_df['diagnosis'] = wis_df['diagnosis'].map({'M': 1, 'B': 0})
    print(wis_df.head(5))
    print(wis_df['diagnosis'].value_counts())
    #f_names = wis_df.columns[wis_df.columns != 'diagnosis']

    X = wis_df.drop(['diagnosis'], axis=1).values[:,:-1]
    y = wis_df['diagnosis'].values
    print(f"X.shape: {X.shape} - y.shape: {y.shape}")

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split,
                         random_state=seed, stratify=y)

    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    return (X_train, y_train), (X_test, y_test)
#from sklearn.datasets import load_breast_cancer
#data = load_breast_cancer()
#train_end = int(data['data'].shape[0] * .9)
#all_data = data['data']
#all_y = data['target']
#all_data, all_y= shuffle(all_data,all_y)
#x = all_data[:train_end]
#y = all_y[:train_end]
#x_val = all_data[train_end+ 1:]
#y_val = all_y[train_end+ 1:]
(x,y),(x_val,y_val) = load_data()
print("shape of x: {}\nshape of y: {}".format(x.shape,y.shape))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


def download_data_file():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

    df_cols = [
        "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
        "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
        "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
        "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]

    print('Downloading data from %s...' % url)
    wis_df = pd.read_csv(url, header=None, names=df_cols, index_col=0)
    wis_df.to_csv(data_file_path)

#defining dataset class
from torch.utils.data import Dataset, DataLoader
class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length
trainset = dataset(x,y)
#DataLoader
trainloader = DataLoader(trainset,batch_size=64,shuffle=True)

#valset = dataset(x_val, y_val)
#valloader = DataLoader(valset,batch_size=32,shuffle=False)
#defining the network
from torch import nn
from torch.nn import functional as F
class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,128)
    self.fc2 = nn.Linear(128,128)
#    self.fc3 = nn.Linear(64,64)
#    self.fc4 = nn.Linear(64,64)
#    self.fc5 = nn.Linear(64,64)
#    self.fc6 = nn.Linear(64,64)
    self.fc7 = nn.Linear(128,128)
    self.fc8 = nn.Linear(128,1)
    self.bias1 = torch.nn.Parameter(torch.rand(1))
    self.bias2 = torch.nn.Parameter(torch.rand(1))
#    self.bias3 = torch.nn.Parameter(torch.rand(1))
#    self.bias4 = torch.nn.Parameter(torch.rand(1))
#    self.bias5 = torch.nn.Parameter(torch.rand(1))
#    self.bias6 = torch.nn.Parameter(torch.rand(1))
    self.bias7 = torch.nn.Parameter(torch.rand(1))
  def forward(self,x):
    x1 = torch.sigmoid(self.fc1(x))
    x2 = torch.sigmoid(self.fc2(x1))
#    x3 = torch.sigmoid(self.fc3(x2))
#    x4 = torch.sigmoid(self.fc4(x3))
#    x5 = torch.sigmoid(self.fc5(x4))
#    x6 = torch.sigmoid(self.fc6(x5))
    x7 = torch.sigmoid(self.fc7(x2))
    x8 = torch.sigmoid(self.fc8(x7))
    return x8,x1,x2,x7, self.bias1,self.bias2, self.bias7
def get_vector(x):
    # 1. Load the image with Pillow library
    # 2. Create a PyTorch Variable with the transformed image
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = np.zeros([x.shape[0],64])
    my_result = None
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_result = o.data.detach().cpu().numpy().squeeze()
        my_embedding[:]  = my_result[:]
        
    # 5. Attach that function to our selected layer
    layer = model._modules.get('fc2')
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    output = model(x)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding, output

#hyper parameters
learning_rate = float(sys.argv[1])
beta = float(sys.argv[2])
epochs = int(sys.argv[3])
grad_clip = float(sys.argv[4])
batch_size = int(sys.argv[5])
# Model , Optimizer, Loss
model = Net(input_shape=x.shape[1])
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True,
                                weight_decay=0.005, momentum=0.9, dampening=0)
loss_fn = nn.BCELoss()
def custom_loss(x,y,bias):
     
    """
     
    :returns
    """
    x_norm = torch.norm(x, dim = 1)
    x_sig = torch.sigmoid(x_norm + bias) 
    y = torch.tensor(y)
    entropy = y * torch.log(x_sig) + (1-y) * torch.log(1 - x_sig)
    ent_sum = -torch.sum(entropy)
    return ent_sum, x_sig
def get_prob(feat_list,biases):
    probs = torch.tensor(np.zeros(feat_list[0].shape[0],))
    for feat, bias in zip(feat_list,biases): 
        x_norm = torch.norm(feat, dim = 1)
        x_sig = torch.sigmoid(x_norm + bias) 
        probs = probs + x_sig
    probs = probs/len(feat_list)
    return probs
def entropy(x):
     
    """
     
    :returns
    """
    x_norm = torch.norm(x, dim = 1)
    x_sig = torch.sigmoid(x_norm + bias.detach().numpy().squeeze()) 
    entropy = -  torch.log(x_sig) -   torch.log(1-x_sig)
    return 1/entropy
def entropy_given_probs(feat_list, biases):
     
    """
     
    :returns
    """
    x_sig = get_prob(feat_list, biases)
    entropy = -  torch.log(x_sig) -   torch.log(1-x_sig)
    return 1/entropy



losses = []
accur = []
soft_accurs = []
layer = model._modules.get('fc2')

neigh1 = KMeans(n_clusters=150)
neigh2 = KMeans(n_clusters=150)
for i in range(epochs):
  for j,(x_train,y_train) in enumerate(trainloader):
    
#    with torch.no_grad(): output , f7,f6,f5,f4,f3,f2,f1,b7,b6,b5,b4,b3,b2,b1 = model(torch.tensor(x,dtype = torch.float32)) 
    with torch.no_grad(): output , f7,f6,f1,b7,b6,b1 = model(torch.tensor(x,dtype = torch.float32)) 
    feat_list = [f7,f6,f1]
#    feat_list = [f7,f6,f5,f4,f3,f2,f1]
    biases = [b7,b6,b1]
    all_entropy = entropy_given_probs(feat_list, biases)
    most_uncertain = random.choices(np.arange(0,x.shape[0]), weights = all_entropy, k = batch_size)
    x_train = torch.tensor(x[most_uncertain], dtype = torch.float32)
    y_train = torch.tensor(y[most_uncertain], dtype=torch.float32)

    #calculate 

    output, f7,f6,f1,b7,b6,b1= model(x_train)
    #calculate loss
#    x_norm = torch.norm(f7, dim = 1)
#    x_sig = torch.sigmoid(x_norm) 
    feat_list = [f7,f6,f1]
    biases = [b7,b6,b1]
#    loss = loss_fn(output,y_train.reshape(-1,1)) + .01 * ent_sum
#    print(f"y_train is {y_train } ")
#    print(f"output is {output } ")
    loss = loss_fn(output.squeeze(),y_train.squeeze()) 
    probb = np.zeros((x_train.shape[0],))
    for feat, bias in zip(feat_list,biases): 
        newloss , newprob = custom_loss(feat, y_train.squeeze(), bias)
        loss += beta * newloss
        probb = probb +  newprob.detach().cpu().numpy()
    probb = probb/len(feat_list)
 
    #accuracy

    
    #backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

  if i%1 == 0:
    with torch.no_grad(): 
        predicted, f7,f6,f1,b7,b6,b1= model(torch.tensor(x_val,dtype=torch.float32))
    feat_list = [f7,f6,f1]
    biases = [b7,b6,b1]
    probss = get_prob(feat_list, biases)
#    print(f"probss is {probss } ")
#    acc = (probb.reshape(-1).detach().numpy().round() == y).mean()
    acc = (probss.reshape(-1).detach().numpy().round() == y_val.squeeze()).mean()
    print(f"acc is {acc } ")
    predicted_rounded = predicted.reshape(-1).detach().numpy().round() 
    soft_acc = (predicted.reshape(-1).detach().numpy().round() == y_val.squeeze()).mean()
#    print(f"predicted is {predicted } ")
    losses.append(loss.detach().numpy())
    accur.append(acc)
    soft_accurs.append(soft_acc)
    print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,soft_acc))
#neigh1.fit(f1, y)
#neigh2.fit(f7,y)
#y1 = neigh1.predict(f1)     
#y2 = neigh2.predict(f7)
#
#y1_acc = metrics.accuracy_score(y1,y)
#print(f"y1_acc is {y1_acc } ")
#y2_acc = metrics.accuracy_score(y2,y)
#print(f"y2_acc is {y2_acc } ")

val_loss = []
accuracy = []
feature_mags = []
predictions = []
model.eval()
for x_train,y_train in trainloader:
    
    #calculate output
    with torch.no_grad():
        output, f7,f6,f1,b7,b6,b1 = model(torch.tensor(x_train,dtype = torch.float32))
#    print(f"output is {output } ")
    #calculate loss

#    print(f"output is {output } ")
    feats = np.linalg.norm(f7.cpu().numpy(), axis = 1)
    feature_mags.append(feats)
    loss = loss_fn(output,y_train.reshape(-1,1))
    #accuracy
    with torch.no_grad():
        predicted, f7,f6,f1,b7,b6,b1= model(torch.tensor(x,dtype=torch.float32))
    predictions.append(output.detach().numpy())
    acc = (predicted.reshape(-1).detach().numpy().round() == y.squeeze()).mean()

    val_loss.append(loss.detach().numpy())
    accuracy.append(acc)
feature_mags = np.concatenate(feature_mags) 
predictions = np.concatenate(predictions)
corcof = np.corrcoef(feature_mags, predictions.squeeze())
print(f"corcof is {corcof } ")

plt.scatter(feature_mags, predictions)
plt.title('Feature Magnitudes versus Likelihoods')
plt.xlabel('Feature Magnitude')
plt.ylabel('Likelihood')
plt.grid()

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,2)
plt.savefig('rhaf_mags_vs_lik_wo.png', bbox_inches='tight')
    
print("Validation: epoch {}\tloss : {}\t accuracy : {}".format(i,np.mean(val_loss),np.mean(accuracy)))
#plotting the loss
plt.figure()
plt.plot(losses)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('rhaf_loss_fig.png')
plt.figure()
#printing the accuracy
plt.plot(accur)
plt.plot(soft_accurs)
plt.title('Accuracy vs Epochs')
plt.xlabel('Accuracy')
plt.ylabel('loss')
ax = plt.gca()
ax.set_ylim([0, 1])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
plt.savefig('rhaf_acc_all_fig.png', bbox_inches='tight')



