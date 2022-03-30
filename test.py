'''
Description: DeepLearning Test
Author: Xu Jiaming
Date: 2022-03-27 20:09:21
LastEditTime: 2022-03-30 22:10:32
LastEditors:  
FilePath: test.py
'''
import numpy as np
a = np.arange(12).reshape(3,4)

b = np.arange(4).reshape((4,1))

c = np.random.random((3,4))
print(a)
print(b)
for i in range(3):
    for j in range(4):
        c[i][j] = a[i][j]*b[j]

print(c)