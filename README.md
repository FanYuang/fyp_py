# A Machine Learning Approach to Building Index Structures(python version with flask)
node.js version is here https://github.com/FanYuang/fyp_node
## Main theme of the project
Compare the setting up time, memory occupation and querying time of different kinds of index/querying method
## How to run
use pip install all packages and then run flask run
## Usage of the routers
### /generatenum_normal
Generate a normal distribution
### /generatenum_uniform
Generate a uniform distribution
### /hashtable_normal
Create a hash table index for the generated normal distribution and test the setting up time, memory occupation and querying time of the method.
### /hashtable_uniform
Create a hash table index for the generated uniform distribution and test the setting up time, memory occupation and querying time of the method.
### /binarysearch_normal
Create a sorted array index for the generated normal distribution and test the setting up time, memory occupation and querying time of binary searching.
### /binarysearch_uniform
Create a sorted array index for the generated uniform distribution and test the setting up time, memory occupation and querying time of binary searching.
### /avl_normal
Create a avl tree index for the generated normal distribution and test the setting up time, memory occupation and querying time of the method.
### /avl_uniform
Create a avl tree index for the generated uniform distribution and test the setting up time, memory occupation and querying time of the method.
### /trick_normal
Create a sorted array index for the generated normal distribution and test the setting up time, memory occupation and querying time of the trick method.
### /trick_uniform
Create a sorted array index for the generated uniform distribution and test the setting up time, memory occupation and querying time of the trick method.
