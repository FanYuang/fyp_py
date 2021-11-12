from flask import Flask
import random
import numpy as np  
import pymongo
import sys
from scipy.stats import norm
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

class TreeNode(object): 
    def __init__(self,_val): 
        self.val = _val 
        self.left = None
        self.right = None
        self.height = 1
        
class AVL_Tree(object): 
    def insert(self, root, val): 
               
        #Simple Bst Insertion:
        if not root: 
            return TreeNode(val) 
        elif val < root.val: 
            root.left = self.insert(root.left, val) 
        else: 
            root.right = self.insert(root.right, val)
       
        # 2)modify the height      
        root.height = 1 + max(self.Height(root.left), self.Height(root.right)) 
        # 3)Get the Balancing Factor
        balance = self.check_Avl(root) 
        # 4)Balance The tree using required set of rotation
        
        #RR Rotation as tree is Left Skewed
        if balance > 1 and val < root.left.val: 
            return self.RR(root) 

        #LL Rotation as tree is Right Skewed
        if balance < -1 and val > root.right.val: 
            return self.LL(root) 
        #RL Rotation as tree is Left then Right Skewed
        if balance > 1 and val > root.left.val: 
            root.left = self.LL(root.left) 
            return self.RR(root) 
        #LR Rotation as tree is Right then Left Skewed
        if balance < -1 and val < root.right.val: 
            root.right = self.RR(root.right) 
            return self.LL(root) 
  
        return root 
     #LL Rotation
    def LL(self, node): 
       
        p = node.right 
        t = p.left
        #Rotations:
        p.left = node 
        node.right = t 
        #modify the heights: 
        node.height = 1 + max(self.Height(node.left), self.Height(node.right)) 
        p.height = 1 + max(self.Height(p.left), self.Height(p.right)) 
   
        return p 
    #LL Rotation
    def RR(self, node): 
  
        p = node.left 
        t = p.right
        #Rotations:
        p.right = node
        node.left = t 
        #modify the heights:
        node.height = 1 + max(self.Height(node.left), self.Height(node.right)) 
        p.height = 1 + max(self.Height(p.left), self.Height(p.right)) 
        return p 
    #Getting the Height
    def Height(self, root): 
        if not root: 
            return 0
  
        return root.height 
    #Getting the Balancing Factor
    def check_Avl(self, root): 
        if not root: 
            return 0
  
        return self.Height(root.left) - self.Height(root.right) 
  
    def preOrder(self, root): 
  
        if not root: 
            return
  
        print("{0} ".format(root.val), end="") 
        self.preOrder(root.left) 
        self.preOrder(root.right) 


def insert_data(_data):
        mytree = AVL_Tree()
        root = None
        for i in _data:
            root = mytree.insert(root,i)
        print("Preorder Traversal of constructed AVL tree is:")
        mytree.preOrder(root)
        print()
        return root

def Search(root,val):
    if (root is None):
        return False
    elif (root.val == val):
        return True
    elif(root.val < val):
        return Search(root.right,val)
    return Search(root.left,val)
    return False


def tricksearch(set, index, target):
    if (index<0 or index>len(set)-1):
        return -1
    elif  (set[index] < target):
        while (set[index]<target):
            index+=1
        if (set[index]==target):
            return index
        else:
            return -1
    elif (set[index]>target):
        while (set[index]<target):
            index+=1
        if (set[index]==target):
            return index
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
        result=dict.get(num)
    
    return "<p>Heiheihei</p>"

@app.route("/hashtable_normal")
def hashtable_normal():
    dict = {}
    for i,num in enumerate(dataset):
        dict[num]=i
    size=sys.getsizeof(dict)
    print(size)
    for num in testset:
        result=dict.get(num)
    return "<p>Heiheihei</p>"

@app.route("/binarysearch_uniform")
def binarysearch_uniform():
    index=sorted(dataset)
    size=sys.getsizeof(index)
    print(size)
    for num in testset:
        result= binarySearch(index, 0, len(index)-1, num)
    return "<p>Heiheihei</p>"


@app.route("/binarysearch_normal")
def binarysearch_normal():
    index=sorted(dataset)
    size=sys.getsizeof(index)
    print(size)
    for num in testset:
        result= binarySearch(index, 0, len(index)-1, num)
    return "<p>Heiheihei</p>"

@app.route("/avl_uniform")
def avl_uniform():
    root = insert_data(testset)
    size=sys.getsizeof(root)
    for num in testset:
        result=Search(root,num)
    return "<p>Heiheihei</p>"


@app.route("/avl_normal")
def avl_normal():
    root = insert_data(testset)
    size=sys.getsizeof(root)
    for num in testset:
        result=Search(root,num)
    return "<p>Heiheihei</p>"

@app.route("/trick_uniform")
def trick_uniform():
    index=sorted(dataset)
    size=sys.getsizeof(index)
    print(size)
    for i,num in enumerate(testset):
        result=tricksearch(index, round(num_uni * (num - low) / (high - low)), num)
    return "<p>Heiheihei</p>"


@app.route("/trick_normal")
def trick_normal():
    index=sorted(dataset)
    size=sys.getsizeof(index)
    print(size)
    for i,num in enumerate(testset):
        result=tricksearch(index, round(num_nor * norm.cdf(num,mean,standard_d)), num)
    return "<p>Heiheihei</p>"