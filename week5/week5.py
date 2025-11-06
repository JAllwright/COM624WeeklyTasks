from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing

header_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
my_data = read_csv("pima_indians_diabetes_2.csv", names=header_names)

print(my_data.shape)

empty_data = my_data[my_data.isna().any(axis=1)]

print(empty_data)

print(my_data.head())
print(my_data.tail())