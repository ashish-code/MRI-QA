import pandas as pd 
import numpy as np 

abide_data_path = '../x_abide.csv'

xabide = pd.read_csv(abide_data_path)

print(xabide.head())
print(xabide.shape)

ds030_data_path = '../x_ds030.csv'
xds030 = pd.read_csv(ds030_data_path)

print(xds030.head())
print(xds030.shape)



xabide_mat = xabide.as_matrix()
# print(xabide_mat)
# print(xabide_mat.shape)

a= xabide_mat[xabide_mat[:,0]==50002][0,1:]
print(a)

# yabide = pd.read_csv('../y_abide.csv')

# yabide_mat = yabide.as_matrix()
# print(yabide.head())
# print(yabide_mat)

# subdf = pd.read_csv('../train_2.csv', header=None)
# subdf.columns = ['sub', 'label']
# # print(subdf.head())

# keys = list(subdf['sub'].values)

# yds030 = pd.read_csv('../y_ds030.csv')
# print(yds030.head())
# yds030_labels = yds030['rater_1'].tolist()

# def count_of(lst, item):
#     count = 0
#     for itm in lst:
#         if itm==item:
#             count += 1
#     return count

# print(count_of(yds030_labels,1))
# print(count_of(yds030_labels,0))
# print(count_of(yds030_labels,-1))



