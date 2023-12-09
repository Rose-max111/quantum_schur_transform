import numpy as np
import math
from numpy import matrix
from numpy import kron
from math import sqrt
from matplotlib import pyplot as plt
import scipy

# Bra and ket are just helper functions that build the appropriate vector out of a 
# bitstring specifying a tensor product
def bra(bitstring):
    return ket(bitstring).getT()
def ket(bitstring):
    # I don't think Python likes padded zeros so this is actually irrelevant
    if not isinstance(bitstring,str):
        bitstring = str(bitstring)
    vec = matrix('1')
    for i in range(len(bitstring)):
        if bitstring[i]=='0':
            vec = kron(vec,matrix('1;0'))
        elif bitstring[i]=='1':
            vec = kron(vec,matrix('0;1'))
    return vec
# A projector matrix from two bitstrings
def ketbra(a,b):
    return ket(a)*bra(b)


# Gives permutation matrix for perm string 'p' on n qubits in n-fold tensor space
def permMatrix(n, p):
    perm = makeDict(n,p)
    mat = np.zeros((2**n,2**n))
    for i in range(2**n):
        basisInt = bin(i)[2:].zfill(n)
        #print(basisInt)
        transformed = permuteString(basisInt, perm)
        #print(transformed)
        #print()
        mat = mat + ket(transformed)*bra(basisInt)
    return mat

# Given a permutation dictionary encoding the desired permutation, 
# it tells you how the permutation would transform
def permuteString(string, permDict):
    newString=''
    permDict = invertDict(permDict)
    for i in range(len(string)):
        newString += string[permDict[i+1]-1]
    return newString

# Not robust at all
# Swaps key and value, but assumes only one value per key
# Used because the action of a permutation on a string actually
# uses the inverse permutation not the original
def invertDict(dictionary):
    newDict = {}
    for i in dictionary.keys():
        newDict[dictionary[i]] = i
    return newDict

# Turns perm string 'p' into a dictionary for easier use
# Keys and values are all ints
def makeDict(n,p):
    key = {}
    for i in range(1,n+1):
        if str(i) in p:
            nextIndex = p.index(str(i))+1
            if nextIndex == len(p):
                nextIndex = 0
            key[i] = int(p[nextIndex])
        else:
            key[i] = i
    return key



schur = matrix([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1/sqrt(3), 0, 0, 0, 0, 2/sqrt(6), 0],
                [0, 1/sqrt(3), 0, 0, 1/sqrt(2), 0, -1/sqrt(6), 0],
                [0,0,1/sqrt(3),0,0,1/sqrt(2),0,1/sqrt(6)],
                [0,1/sqrt(3),0,0,-1/sqrt(2),0,-1/sqrt(6),0],
                [0,0,1/sqrt(3),0,0,-1/sqrt(2),0,1/sqrt(6)],
                [0,0,1/sqrt(3),0,0,0,0,-2/sqrt(6)],
                [0,0,0,1,0,0,0,0]
               ])

# mat12 = schur**(-1) * permMatrix(3, "12") * schur   
# print(mat12.round(decimals=2))

y = 0.95 / 0.02
for n in range(100):
    c = (3/2) ** n
    if(c >= y):
        print(n)
        break