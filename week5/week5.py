from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing

header_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
my_data = read_csv("pima_indians_diabetes_2.csv", names=header_names)

print('The dimension of the initial dataframe is ', my_data.shape)

new_data = my_data.dropna()

print('\nThe dimension of the new dataframe with dropped null values is ', new_data.shape)

print('\nThe last 5 rows of the new dataframe are ', new_data.tail(5))