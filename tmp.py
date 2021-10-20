from numpy import linalg as la
import numpy as np

import sympy

import scipy

from PIL import Image
import matplotlib.pyplot as plt

from itertools import permutations

matrix = [[1, 5, 7, 6, 1],
          [2, 1, 10, 4, 4],
          [3, 6, 7, 5, 2]]
matrix = np.array(matrix)

U, s, Vtrans = la.svd(matrix)

Sigma = np.zeros(np.shape(matrix))
Sigma[:len(s), :len(s)] = np.diag(s)

x = sympy.symbols('x')
res = (112-x)*(137-x)*(123-x) + 110*105*114*2 - 114*114*(137-x) - 110*110*(112-x) - 105*105*(123-x)
print(sympy.solve(res, x))
print(matrix.dot(np.transpose(matrix)))
print(U)
print(Sigma)
print(Vtrans)


result = U.dot(Sigma.dot(Vtrans))
print("U*Sigma*Vtrans result: \n", result)


A = [[1, 1, 1],
     [1, 2, 4],
     [1, 3, 9]]
A = np.array(A)

B = [2, 3, 5]
B = np.array(B)

print(la.inv(A).dot(B))
print(la.solve(A, B))
x, y, z = sympy.symbols("x y z")
print(sympy.solve([x+y+z-2, x+2*y+4*z-3, x+3*y+9*z-5], [x, y, z]))


def approxSVD(data, percent):
    U, s, Vtrans = la.svd(data)
    Sigma = np.zeros(np.shape(data))
    Sigma[:len(s), :len(s)] = np.diag(s)
    k = int(percent * len(s))
    res = U[:, :k].dot(Sigma[:k, :k].dot(Vtrans[:k, :]))
    res[res < 0] = 0
    res[res > 255] = 255

    return np.rint(res).astype("uint8")


def approxSVD2(data, percent):
    U, s, Vtrans = la.svd(data)
    Sigma = np.zeros(np.shape(data))
    Sigma[:len(s), :len(s)] = np.diag(s)
    count = int(percent * sum(s))
    nowsum = 0
    k = -1
    while nowsum < count:
        k += 1
        nowsum += s[k]
    res = U[:, :k].dot(Sigma[:k, :k].dot(Vtrans[:k, :]))
    res[res < 0] = 0
    res[res > 255] = 255

    return np.rint(res).astype("uint8")


img = Image.open("D:\\data\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000129.jpg", 'r')
img = np.array(img)


def rebuild(percent):
    res = [[] for _ in range(6)]
    for i in range(3):
        res[i] = approxSVD(img[:, :, i], percent)
        res[i+3] = approxSVD2(img[:, :, i], percent)
    outimg1 = np.stack((res[0], res[1], res[2]), 2)
    outimg2 = np.stack((res[3], res[4], res[5]), 2)

    outimg1 = Image.fromarray(outimg1)
    outimg2 = Image.fromarray(outimg2)
    return outimg1, outimg2


plt.figure()
k = 1
for i in range(20, 120, 20):
    outimg1, outimg2 = rebuild(float(i/100))
    plt.subplot(2, 5, k)
    plt.imshow(outimg1)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 5, k+5)
    plt.imshow(outimg2)
    plt.xticks([])
    plt.yticks([])

    k += 1
