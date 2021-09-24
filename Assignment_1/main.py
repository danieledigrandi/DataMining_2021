# Data mining project 2021 - assignment 1
import pandas as pd
import numpy

# importing credit file
credit = pd.read_csv('./Assignment_1/data/credit.txt')

# importing pima file
pima = pd.read_csv('./Assignment_1/data/pima.txt')

# importing eclipse for training
training = pd.read_csv('./Assignment_1/data/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv', sep=';')

# importing eclipse for testing
testing = pd.read_csv('./Assignment_1/data/promise-2_0a-packages-csv/eclipse-metrics-packages-3.0.csv', sep=';')

print(credit)