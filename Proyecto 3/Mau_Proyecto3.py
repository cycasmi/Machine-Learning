#Hector Mauricio Gonzalez Coello
#A01328258
#Machine Learning
#ITESM AUGUST-DEC 2017

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math

def readFiles():
	x =[]
	y=[]
	f=open("data.txt")
	lines=f.readlines()
	for line in lines:
		x.append([float(line.split(',')[0]), float(line.split(',')[1])])
		y.append(float(line.split(',')[2]))
	f.close()
	return x, y

def graficaDatos(X,y,theta):
    print (theta)
    plt.plot(theta)
    plt.show()

def sigmoidal(z):
    divisor = (1.0 + (np.exp((-1.0)*z)))
    sigmo = 1.0 / divisor
    return sigmo

def funcionCosto(theta,X,y):
    sumErr = 0
    newX = 0
    ysize = len(y)
    J = 0
    tmp = 0
    for i in range(len(X)):
        tmp = X[i] * theta
        hipo = sigmoidal(tmp)
        first = -y[i] * np.log(hipo)
        second = (1-y[i]) * np.log(1-hipo)
        J += first - second
    J *= (1/len(y))
    
    print ("J: ")
    print (J)

    grad =  np.zeros(len(theta))
    for i in range(len(theta)):
        newX += X[i]*theta
    h=sigmoidal(newX)

    for i in range(len(grad)):
        grad[i] = (1/len(X)) * sum(((h.transpose().dot(x0))-y0)*x0[i] for (x0,y0) in zip(X,y) )
    return J, grad

def aprende(theta,X,y,iteraciones):
    theta = np.zeros(np.shape(X[0]))
    costo = funcionCosto(theta,X,y)
    for j in range(len(theta)):
        tmp = [theta[j] - costo]
        theta = np.append(theta, tmp)
    return theta

def predice(theta,X):
    predict = np.zeros(len(X))
    for i in range(len(X)):
        if (sigmoidal(X[i].dot(theta))>0.5):
            predict[i] =1
        else:
            predict[i]=0
    return predict

def main():
    x = []
    y = []
    x, y = readFiles()
    theta=np.zeros(np.shape(x))
    theta = aprende(theta,x,y,1000)
    graficaDatos(x,y,theta)

main()