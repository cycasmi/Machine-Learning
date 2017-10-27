# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:32:22 2017

@author: Cynthia B. Castillo Millán
Matrícula: A01374530
"""
import  numpy as np
import matplotlib.pyplot as plt

def leerDatos(xValues, yValues, file):
    data = open(file,"r")
    for line in data.readlines():
        line.split(",")
        xValues.append(float(line.split(",")[0]))
        yValues.append(float(line.split(",")[1]))        
    return

def graficaDatos(X, y, theta):
    t = np.arange(1, max(X), 1) #values to trace the line
    
    #plot in blue dots the data and in a red line for the function
    plt.xlabel('Pies cuadrados')
    plt.ylabel('Costo (USD, 10,000)')
    plt.title('Costo de casas')
    plt.plot(X, y, 'bo',t, theta[0]+t*theta[1], 'r-')
    plt.show()
    return 

def gradienteDescendente(X, y, theta, alpha, iteraciones):
    
    temp_theta = [theta[0], theta[1]]
    cost = calculaCosto(X, y, temp_theta) #
    print ("Costo inicial: "+repr(cost[0]) + "," + repr(cost[1])+ "\n\n")
    
    for i in range(0,iteraciones):
        cost = calculaCosto(X, y, temp_theta)
        theta[0] = temp_theta[0] - alpha*(1/len(X))*cost[0]
        theta[1] = temp_theta[1] - alpha*(1/len(X))*cost[1]
        temp_theta = [theta[0], theta[1]]
    
    print ("Costo final: " + repr(cost[0]) + "," + repr(cost[1]) + "\n\n")  
    return theta

def calculaCosto(X, y, theta):
    sum0 = 0
    sum1 = 1
    
    #adding the errors, the cost function
    for i, val in enumerate(X):
        sum0 = sum0 + (theta[0]+theta[1]*X[i])-y[i] 
        sum1= sum1 +((theta[0]+theta[1]*X[i])-y[i])*X[i]
    
    sums = [sum0, sum1]
    return sums

def main():
    xValues = []
    yValues = []
    dataFile = "data.txt"
    theta = [0.0,0.0]
    alpha = 0.01
    iteraciones = 1500
    
    leerDatos(xValues, yValues, dataFile)
    theta = gradienteDescendente(xValues, yValues, theta, alpha, iteraciones)
    graficaDatos(xValues, yValues, theta)
    return

main()
#print("Prediccion1= [1, 3.5] * theta = " + repr(format(theta[0] + theta[1]*3.5, '.4f')))
#print("Prediccion1= [1, 7] * theta = " + repr(format(theta[0] + theta[1]*7, '.4f')))
