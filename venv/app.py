from flask import Flask
import random
import numpy as np  
import pymongo
import sys
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
num_nor = 20000
low=0
high=10000
num_uni = 10000

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

def binarySearch (arr, l, r, x): 
  
    
    if r >= l: 
  
        mid = int(l + (r - l)/2)
  
        
        if arr[mid] == x: 
            return mid 
          
       
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
  
       
        else: 
            return binarySearch(arr, mid+1, r, x) 
  
    else: 
        
        return -1

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/generatenum_normal")
def generate_normal():
    global dataset 
    dataset = generatenum_normal(mean,standard_d, num_nor)
    global testset
    testset=generatenum_normal(mean, standard_d, num_nor)
    print(dataset)
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
    dict = {}
    for i,num in enumerate(dataset):
        dict[num]=i
    size=sys.getsizeof(dict)
    print(size)
    for num in testset:
        index=dict.get(num)
    
    return "<p>Heiheihei</p>"

@app.route("/hashtable_normal")
def hashtable_normal():
    dict = {}
    for i,num in enumerate(dataset):
        dict[num]=i
    size=sys.getsizeof(dict)
    print(size)
    for num in testset:
        index=dict.get(num)
    return "<p>Heiheihei</p>"

@app.route("/binarysearch_uniform")
def binarysearch_uniform():
    
    return "<p>Heiheihei</p>"


@app.route("/binarysearch_normal")
def binarysearch_normal():
    
    return "<p>Heiheihei</p>"