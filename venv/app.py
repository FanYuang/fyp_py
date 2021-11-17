from flask import Flask
import random
import numpy as np  
import pymongo
import sys
from scipy.stats import norm
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import time
import psutil
import os

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["fyp"]
mycol = mydb["data"]
# mydict = { "name": "RUNOOB", "alexa": "10000", "url": "https://www.runoob.com" }
 
# x = mycol.insert_one(mydict) 
app = Flask(__name__)

dataset=[]
testset=[]
mean = 0
standard_d = 1000
num_nor = 10000
low=0
high=100000
num_uni =100000

class MyDict(object):
    def __init__(self, size=99999):
        self.hash_list = [list() for _ in range(size)]
        self.size = size

def __setitem__(self, key, value):
    hashed_key = hash(key) % self.size
    for item in self.hash_list[hashed_key]:
        if item[0] == key:
            item[1] = value
            break
    else:
        self.hash_list[hashed_key].append([key, value])

def __getitem__(self, key):
    for item in self.hash_list[hash(key) % self.size]:
        if item[0] == key:
            return item[1]
    return -1

def generatenum_uniform(low, high, num_uni):
    arr=np.random.uniform(low, high, num_uni)
    array=[]
    for num in arr:
        array.append(round(num))
    return array

def generatenum_normal(mean, standard_d, num_nor):
    arr=np.random.normal(mean,standard_d,num_nor)
    array=[]
    for num in arr:
        array.append(round(num))
    return array

# def binarySearch (arr, l, r, x): 
#     if r >= l: 
#         mid = int(l + (r - l)/2)
#         if arr[mid] == x: 
#             return mid 
#         elif arr[mid] > x: 
#             return binarySearch(arr, l, mid-1, x)
#         else: 
#             return binarySearch(arr, mid+1, r, x) 
#     else: 
#         return -1

def binarySearch(OrderedList, left, right, key):
    while left <= right:
        mid = (left + right) // 2
        if key == OrderedList[mid]:
            return mid
        elif key > OrderedList[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return None


class TreeNode(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        self.height = 0
class AVLTree(object):
    def __init__(self):
        self.root = None
    def find(self, key):
        if not self.root:
            return None
        else:
            return self._find(key, self.root)
    def _find(self, key, node):
        if not node:
            return None
        elif key < node.data:
            return self._find(key, node.left)
        elif key > node.data:
            return self._find(key, node.right)
        else:
            return node
    def findMin(self):
        if self.root is None:
            return None
        else:
            return self._findMin(self.root)
    def _findMin(self, node):
        if node.left:
            return self._findMin(node.left)
        else:
            return node
    def findMax(self):
        if self.root is None:
            return None
        else:
            return self._findMax(self.root)
    def _findMax(self, node):
        if node.right:
            return self._findMax(node.right)
        else:
            return node
    def height(self, node):
        if node is None:
            return -1
        else:
            return node.height
    #在node节点的左孩子k1的左子树添加了新节点，左旋转
    def singleLeftRotate(self, node):
        k1 = node.left
        node.left = k1.right
        k1.right = node
        node.height = max(self.height(node.right), self.height(node.left)) + 1
        k1.height = max(self.height(k1.left), node.height) + 1
        return k1
    #在node节点的右孩子k1的右子树添加了新节点，右旋转
    def singleRightRotate(self, node):
        k1 = node.right
        node.right = k1.left
        k1.left = node
        node.height = max(self.height(node.right), self.height(node.left)) + 1
        k1.height = max(self.height(k1.right), node.height) + 1
        return k1
    #在node节点的左孩子的右子树添加了新节点，先左后右
    def doubleRightRotate(self, node):
        node.right = self.singleLeftRotate(node.right)
        return self.singleRightRotate(node)
    #在node节点的右孩子的左子树添加了新节点,先右后左
    def doubleLeftRotate(self, node):
        node.left = self.singleRightRotate(node.left)
        return self.singleLeftRotate(node)
    def insert(self, key):
        if not self.root:
            self.root = TreeNode(key)
        else:
            self.root = self._insert(key, self.root)
    def _insert(self, key, node):
        if node is None:
            node = TreeNode(key)
        elif key < node.data:
            node.left = self._insert(key, node.left)
            if (self.height(node.left) - self.height(node.right)) == 2:
                if key < node.left.data:
                    node = self.singleLeftRotate(node)
                else:
                    node = self.doubleLeftRotate(node)
        elif key > node.data:
            node.right = self._insert(key, node.right)
            if (self.height(node.right) - self.height(node.left)) == 2:
                if key > node.right.data:
                    node = self.singleRightRotate(node)
                else:
                    node = self.doubleRightRotate(node)
        node.height = max(self.height(node.right), self.height(node.left)) + 1
        return node
    def delete(self, key):
        if self.root is None:
            raise KeyError('Error,empty tree')
        else:
            self.root = self._delete(key, self.root)
    def _delete(self, key, node):
        if node is None:
            raise KeyError('Error,key not in tree')
        elif key < node.data:
            node.left = self._delete(key, node.left)
            if (self.height(node.right) - self.height(node.left)) == 2:
                if self.height(node.right.right) >= self.height(node.right.left):
                    node = self.singleRightRotate(node)
                else:
                    node = self.doubleRightRotate(node)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        elif key > node.data:
            node.right = self._delete(key, node.right)
            if (self.height(node.left) - self.height(node.right)) == 2:
                if self.height(node.left.left) >= self.height(node.left.right):
                    node = self.singleLeftRotate(node)
                else:
                    node = self.doubleLeftRotate(node)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        elif node.left and node.right:
            if node.left.height <= node.right.height:
                minNode = self._findMin(node.right)
                node.key = minNode.key
                node.right = self._delete(node.key, node.right)
            else:
                maxNode = self._findMax(node.left)
                node.key = maxNode.key
                node.left = self._delete(node.key, node.left)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        else:
            if node.right:
                node = node.right
            else:
                node = node.left
        return node


def tricksearch(sett, index, target):
    if (index<0 or index>(len(sett)-1)):
        # if (sett[0]>target or sett[len(sett)-1]<target):
        return -1
        # elif (index<0):
        #     tricksearch(sett,0,target)
        # elif (index>len(sett)-1):
        #     tricksearch(sett,len(sett)-1,target)
    elif  (sett[index] < target):
        while (sett[index]<target):
            if (index<len(sett)-1):
                index+=1
            else:
                return -1
        # if (sett.index(target)):
        #     print(index,sett.index(target),sett[index],"<")
        if (sett[index]==target):
            return index
        else:
            return -1
    elif (sett[index]>target):
        while (sett[index]>target):
            if (index>0):
                index-=1
            else:
                return -1
        # if (sett.index(target)):
        #     print(index,sett.index(target),sett[index],">")
        if (sett[index]==target):
            return index
        else:
            return -1
    elif (sett[index]==target):
        return index



    


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/generatenum_normal")
def generate_normal():
    global dataset 
    dataset = generatenum_normal(mean,standard_d, num_nor)
    global testset
    testset=generatenum_normal(mean, standard_d, num_nor)
    # print(dataset)
    return "<p>Heiheihei</p>"

@app.route("/generatenum_uniform")
def generate_uniform():
    global dataset 
    dataset= generatenum_uniform(low, high, num_uni)
    global testset
    testset= generatenum_uniform(low, high, num_uni)
    return "<p>Heiheihei</p>"

@app.route("/hashtable_uniform")
def hashtable_uniform():
    # dict = {}
    dict=MyDict()
    start=time.time()
    for i,num in enumerate(dataset):
        __setitem__(dict, num, i)
     
    end=time.time()
    size=sys.getsizeof(dict)
    setup=end-start
    start=time.time()
    for num in testset:
        result=__getitem__(dict, num)
      
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"

@app.route("/hashtable_normal")
def hashtable_normal():
    dict=MyDict()
    start=time.time()
    for i,num in enumerate(dataset):
        __setitem__(dict, num, i)
     
    end=time.time()
    size=sys.getsizeof(dict)
    setup=end-start
    start=time.time()
    for num in testset:
        result=__getitem__(dict, num)
     
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"

@app.route("/binarysearch_uniform")
def binarysearch_uniform():
    start=time.time()
    index=sorted(dataset)
    end=time.time()
    setup=end-start
    size=sys.getsizeof(index)
    start=time.time()
    for num in testset:
        result= binarySearch(index, 0, len(index)-1, num)
       
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"


@app.route("/binarysearch_normal")
def binarysearch_normal():
    start=time.time()
    index=sorted(dataset)
    end=time.time()
    setup=end-start
    size=sys.getsizeof(index)
    start=time.time()
    for num in testset:
        result= binarySearch(index, 0, len(index)-1, num)
      
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"

@app.route("/avl_uniform")
def avl_uniform():
    start=time.time()
    root=AVLTree()
    for num in dataset:
        root.insert(num)

    end=time.time()
    setup=end-start
    size=sys.getsizeof(root)
    start=time.time()
    # print(root)
    for num in testset:
        result=root.find(num)
        # print(result)
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"


@app.route("/avl_normal")
def avl_normal():
    start=time.time()
    root=AVLTree()
    for num in dataset:
        root.insert(num)

    end=time.time()
    setup=end-start
    size=sys.getsizeof(root)
    start=time.time()
    # print(root)
    for num in testset:
        result=root.find(num)
        # print(result)
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"

@app.route("/trick_uniform")
def trick_uniform():
    start=time.time()
    index=sorted(dataset)
    
    end=time.time()
    setup=end-start
    size=sys.getsizeof(index)
    start=time.time()
    mean_var=0
    a=0
    l=len(index)
    for i,num in enumerate(testset):
        result=tricksearch(index,round(num_uni * (num - low) / (high - low)),num)
        # result= binarySearch(index, max(round(num_uni * (num - low) / (high - low)-l/500),0), min(len(index)-1,round(num_uni * (num - low) / (high - low)+l/500)), num)
        # if (result!=-1):
        #     a=a+1
        #     mean_var=mean_var+abs(round(num_uni * (num - low) / (high - low))-result)
        #     print(round(num_uni * (num - low) / (high - low)),result)
    end =time.time()
    # mean_var=mean_var/a
    # print(mean_var,a)
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"


@app.route("/trick_normal")
def trick_normal():
    start=time.time()
    index=sorted(dataset)
    end=time.time()
    setup=end-start
    size=sys.getsizeof(index)
    start=time.time()
    for i,num in enumerate(testset):
        result=tricksearch(index,round(num_nor * norm.cdf(num,mean,standard_d)),num)
        # print(round(num_nor * norm.cdf(num,mean,standard_d)),num,result)
        # result= binarySearch(index, max(round(num_nor * norm.cdf(num,mean,standard_d)-len(index)/100),0), min(len(index)-1,round(num_nor * norm.cdf(num,mean,standard_d)+len(index)/100)), num)
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # self.hidden=nn.Linear(n_feature,n_hidden)
        # self.predict=nn.Linear(n_hidden,n_output)
        self.fc1=nn.Linear(1,16)
        self.act1=nn.ReLU()
        self.fc2=nn.Linear(16,1)
        # self.act2=nn.Sigmoid()
        # self.fc3=nn.Linear(2,1)
        # self.act3=nn.Sigmoid()
    def forward(self,x):
        x=self.fc1(x)
        x=self.act1(x)
        x=self.fc2(x)
        return(x)

@app.route("/ml_uniform")
def ml_uniform():
    data=sorted(dataset)
    test=[]
    for num in data:
        test.append(num/data[len(dataset)-1])
    tar=[]
    for i in range(len(dataset)):
        tar.append(i/(len(dataset)-1))
    start=time.time()
    input=torch.tensor(test, dtype=torch.float)
    input=input.unsqueeze(1)
    target=torch.tensor(tar,  dtype=torch.float)
    target=target.unsqueeze(1)
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for i in range(1000):
        output = net(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(list(net.parameters()))
        # print(output)
        # print(loss,i)
    end=time.time()
    setup=end-start
    # print(net)
    size=sys.getsizeof(net)
    testin=[]
    for num in testset:
        testin.append(num/data[len(dataset)-1])
    test_input=torch.tensor(testin, dtype=torch.float)
    test_input=test_input.unsqueeze(1)
    output=net(test_input)
    testout=output.detach().numpy()
    start=time.time()
    for i,num in enumerate(testout):
        
        result= binarySearch(data, max(0,round(num[0]*(len(dataset)-1)-len(data)/100)), min(len(data)-1,round(num[0]*(len(data)-1)+len(data)/100)),testset[i])
        
        # print(result,round(num[0]*(len(dataset)-1)),testset[i],i)
    end=time.time()
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"


@app.route("/ml_normal")
def ml_normal():
    data=sorted(dataset)
    test=[]
    for num in data:
        test.append(num/data[len(dataset)-1])
    tar=[]
    for i in range(len(dataset)):
        tar.append(i/(len(dataset)-1))
    start=time.time()
    input=torch.tensor(test, dtype=torch.float)
    input=input.unsqueeze(1)
    target=torch.tensor(tar,  dtype=torch.float)
    target=target.unsqueeze(1)
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for i in range(1000):
        output = net(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(list(net.parameters()))
        # print(output)
        # print(loss,i)
    end=time.time()
    setup=end-start
    size=sys.getsizeof(net)
    testin=[]
    for num in testset:
        testin.append(num/data[len(dataset)-1])
    test_input=torch.tensor(testin, dtype=torch.float)
    test_input=test_input.unsqueeze(1)
    output=net(test_input)
    testout=output.detach().numpy()
    mean_var=0
    a=0
    l=len(data)
    bin=[]
    for i,num in enumerate(testout):
        # if (testset[i]==)
        # result=tricksearch(data,round(num[0]*(len(dataset)-1)),testset[i])
        arr=[]
        arr.append(max(0,round(num[0]*(l-1)-l/500)))
        arr.append(min(l-1,round(num[0]*(l-1)+l/500)))
        arr.append(testset[i])
        arr.append(data[arr[0]:arr[1]])
        arr.append(len(arr[3])-1)
        bin.append(arr)
 
    start=time.time()
    for i,num in enumerate(testout):
        # if (testset[i]==)
        result=tricksearch(data,round(num[0]*(len(dataset)-1)),testset[i])
        # result= binarySearch(num[3], 0,num[4],num[2])
        # print(result,max(0,round(num[0]*(len(dataset)-1)-len(data)/500)), min(len(data)-1,round(num[0]*(len(dataset)-1)+len(data)/500)),i)
     
        # if (result!=-1):
        #     a=a+1
        #     mean_var=mean_var+abs(round(num[0]*(len(dataset)-1))-result)
        #     print(round(num[0]*(len(dataset)-1)),result)
    end =time.time()
    # mean_var=mean_var/a
    # print(mean_var,a)
    query=end-start
    print(size,setup,query)
    return "<p>Heiheihei</p>"
