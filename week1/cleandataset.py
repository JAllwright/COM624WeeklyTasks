import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('titanic_samples.csv')

df.info()

df.isnull().sum()

df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

df.fillna({'Age': df['Age'].median()}, inplace=True)

df.to_csv('titanic_cleaned.csv', index=False)
