# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:04:39 2019

@author: Christo
"""

import numpy as np
import random
import operator
import math

def initializeMembershipMatrix(x,n_topics):
    membership_mat = list()
    for i in range(len(x)):
        random_num_list = [random.random() for i in range(n_topics)] 
        summation = sum(random_num_list)
        temp_list = [x1/summation for x1 in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat

def initializeWeight(x):
    Weight= [1 for i in range(len(x))] 
    return Weight 

def calculateClusterCenter(membership_mat,W,sx,m,n_topics):
    cluster_mem_val = list(zip(*membership_mat))   
    cluster_centers = list()
    for j in range(n_topics):
        x = list(cluster_mem_val[j])
        xraised=[e ** m for e in x]
        xraised_mul_W=[a*b for a,b in list(zip(xraised,W))]
        denominator=sum(xraised_mul_W)
        temp_num = list()
        for i in range(len(sx)):
            data_point = sx[i]
            prod = [xraised_mul_W[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num))) 
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

def updatemembershipvalue(U,C,sx,m,n_topics):
    alpha=float(1/(m-1))
    for i in range(n_topics):
        for j in range(len(sx)):
            x=sx[j]
            numerator=[(a-b)**2 for a,b in list(zip(x,C[i]))]
            num=sum(numerator)
            dis=[map(operator.sub,x,C[k1]) for k1 in range(n_topics)]
            denominator=[map(lambda x: x**2, dis[j1]) for j1 in range(n_topics)]
            den=[sum(denominator[k1]) for k1 in range(n_topics)]
            res=sum([math.pow(float(num/den[k1]),alpha) for k1 in range(n_topics)])
            U[j][i]=float(1/res)
    return U  

def updateweight(U,sx,w,n_topics):
    W=list()
    u=list(zip(*U))
    for i in range(n_topics):
        u1=sum([a*b for a,b in list(zip(u[i],w))])
        W.append(u1)
    return W

def WFCM(sx,W,U,C,max_iter,m,n_topics):
    i=0
    while(i<=max_iter):     
        U=updatemembershipvalue(U,C,sx,m,n_topics)
        C=calculateClusterCenter(U,W,sx,m,n_topics)
        i+=1
    return C,U      

def spFCM(df,chunk,n_topics,max_iter,m):
    X_sampled=list()
    for j in range(chunk):
        l=int(len(df)/chunk)
        x=list()
        for i in range(j*l,min(j*l+l,int(len(df)))):
            data_chunk=list(df[i])
            x.append(data_chunk)
        X_sampled.append(x)    
    W=initializeWeight(X_sampled[0])
    U=initializeMembershipMatrix(X_sampled[0],n_topics)
    center=calculateClusterCenter(U,W,X_sampled[0],m,n_topics)
    C,U=WFCM(X_sampled[0],W,U,center,max_iter,m,n_topics)
    X=X_sampled[0]
    for j in range(1,chunk):
        W1=updateweight(U,X,W,n_topics)
        X=center+ X_sampled[j] 
        W2=initializeWeight(X)
        W=W1+W2
        U=initializeMembershipMatrix(X,n_topics)
        center=calculateClusterCenter(U,W,X,m,n_topics)
        C,U=WFCM(X,W,U,center,max_iter,m,n_topics)
    return C,U