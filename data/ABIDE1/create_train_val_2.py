"""
Process the ABIDE 1 annotations provided by MRIQC to create training, validation and testing sets.

Author: ashish gupta
Email: ashishagupta@gmail.com
"""

import pandas as pd 
import numpy as np 
import os
import sys

data_file_path = '../../y_abide.csv'
y_abide = pd.read_csv(data_file_path)

y_val = y_abide[y_abide['site'].isin(['NYU', 'LEUVEN'])]
# print(y_test)

y_train = y_abide[~y_abide['site'].isin(['NYU', 'LEUVEN'])]
# print(y_train_val)

# print(y_test.head())

# y_test_consensus = y_test[['subject_id', 'site']]
# y_test_accept_index = []
# test_consensus_rating = []
# for index, row in y_test.iterrows():
#     if len(pd.Series(row[['rater_1', 'rater_2', 'rater_3']]).dropna().unique())==1:
#         y_test_accept_index.append(index)
#         test_consensus_rating.append(pd.Series(row[['rater_1', 'rater_2', 'rater_3']]).dropna().unique()[0])

# y_test_consensus = y_test_consensus.ix[y_test_accept_index]
# # print(y_test_consensus.head())

# y_test_consensus['rating'] = test_consensus_rating
# # print(y_test_consensus.head())

# print(len([i for i in test_consensus_rating if i==0]))
# print(len([i for i in test_consensus_rating if i==1]))
# print(len([i for i in test_consensus_rating if i==-1]))


# y_test_negative = y_test[['subject_id', 'site']]
# y_test_negative_index = []
# test_consensus_rating = []
# for index, row in y_test.iterrows():
#     if -1 in row[['rater_1', 'rater_2', 'rater_3']].values:
        
#         y_test_negative_index.append(index)
#         test_consensus_rating.append(pd.Series(row[['rater_1', 'rater_2', 'rater_3']]).dropna().unique()[0])


# y_test_negative = y_test.ix[y_test_negative_index]
# print(y_test_negative)
# print(len(y_test_negative_index))


"""
We will have train and validation sets. There are two class labels for subjects.
If there is even one unacceptable rating (-1.0) among the three raters then the subject is considered unacceptable.
If there is a concensus positive rating (+1.0) among the three raters then the subject is acceptable.
The intention is to increase the number of unacceptable samples, which are inherently low in the ABIDE 1 dataset.
"""

# validation set
y_val_pos_idx = []
y_val_neg_idx = []

for index, row in y_val.iterrows():
    _temp = pd.Series(row[['rater_1', 'rater_2', 'rater_3']]).dropna().unique()
    if len(_temp)==1 and _temp[0]==1.0:
        y_val_pos_idx.append(index)
    if -1 in row[['rater_1', 'rater_2', 'rater_3']].values:
        y_val_neg_idx.append(index)

y_val_pos = y_val.ix[y_val_pos_idx]
y_val_neg = y_val.ix[y_val_neg_idx]

# training set
y_train_pos_idx = []
y_train_neg_idx = []

for index, row in y_train.iterrows():
    _temp = pd.Series(row[['rater_1', 'rater_2', 'rater_3']]).dropna().unique()
    if len(_temp)==1 and _temp[0]==1.0:
        y_train_pos_idx.append(index)
    if -1 in row[['rater_1', 'rater_2', 'rater_3']].values:
        y_train_neg_idx.append(index)

y_train_pos = y_train.ix[y_train_pos_idx]
y_train_neg = y_train.ix[y_train_neg_idx]

# debug
print(y_val_pos.shape)
print(y_val_neg.shape)
print(y_train_pos.shape)
print(y_train_neg.shape)

# train file
print(y_train_pos.head())
train_list_file_path = 'train_2.csv'
data_root_dir = '/mnt/Depo/Datasets/ABIDE1/RawDataBIDS/'
with open(train_list_file_path, 'w') as f:
    for index, row in y_train_pos.iterrows():
        sub_id = int(row['subject_id'])
        # sub_id = str(row['subject_id']).zfill(7) # zero padding the string with leading zeros
        site_id = row['site']
        label = 1
        nii_file_path = f'{data_root_dir}{site_id}/sub-{sub_id}/anat/sub-{sub_id}_T1w.nii.gz'
        f.write(f'{sub_id},{label}\n')

    for index, row in y_train_neg.iterrows():
        sub_id = int(row['subject_id'])
        # sub_id = str(row['subject_id']).zfill(7)
        site_id = row['site']
        label = -1
        nii_file_path = f'{data_root_dir}{site_id}/sub-{sub_id}/anat/sub-{sub_id}_T1w.nii.gz'
        f.write(f'{sub_id},{label}\n')

# validation file
print(y_val_pos.head())
val_list_file_path = 'val_2.csv'
with open(val_list_file_path, 'w') as f:
    for index, row in y_val_pos.iterrows():
        # sub_id = str(row['subject_id']).zfill(7)
        sub_id = int(row['subject_id'])
        site_id = row['site']
        label = 1
        nii_file_path = f'{data_root_dir}{site_id}/sub-{sub_id}/anat/sub-{sub_id}_T1w.nii.gz'
        f.write(f'{sub_id},{label}\n')
    for index, row in y_val_neg.iterrows():
        # sub_id = str(row['subject_id']).zfill(7)
        sub_id = int(row['subject_id'])
        site_id = row['site']
        label = -1
        nii_file_path = f'{data_root_dir}{site_id}/sub-{sub_id}/anat/sub-{sub_id}_T1w.nii.gz'
        f.write(f'{sub_id},{label}\n')

                







