import pandas as pd 
import numpy as np 


def create_data_file(label_file, out_file):
    data_file = './x_abide.csv'
    # read the label csv file
    label_df = pd.read_csv(label_file, header=None)
    # assign column names
    label_df.columns = ['subject_id', 'label']
    # set the column with subject id as index of dataframe
    label_df = label_df.set_index('subject_id')
    # change the label for negative class from -1 to 0
    label_df = label_df.replace(-1, 0)
    # read the data csv file
    data_df = pd.read_csv(data_file)
    # set the column with subject id as index of dataframe
    data_df = data_df.set_index('subject_id')
    # slice the training data from the entire data corpus
    data_df_select = data_df[data_df.index.isin(label_df.index)]
    # ensure the bijective mapping between labels and data in case of missing data
    label_df = label_df[label_df.index.isin(data_df_select.index)]
    # sort the dataframe by the subject_id as index
    label_df.sort_index(inplace=True)
    data_df_select.sort_index(inplace=True)
    # concatenate the data with the label
    data_df_cat = pd.concat([data_df_select, label_df], axis=1)
    # debug
    print(data_df_cat.shape)
    # write dataframe to csv without index and header
    data_df_cat.to_csv(out_file, sep=',', header=False, index=False)



def create_train_val_files(label_files, out_files):
    """create training and validation data files for abide 1
    
    Arguments:
        label_files {[type]} -- [description]
        out_files {[type]} -- [description]
    """
    for i in range(len(label_files)):
        label_file = label_files[i]
        out_file = out_files[i]
        create_data_file(label_file, out_file)
        print(f'created {out_file}')


def create_test_file():
    """test data file for DS030 dataset
    """
    data_file = './x_ds030.csv'
    label_file = './y_ds030.csv'
    out_file = './ds030_test.csv'

    # read the data csv file
    data_df = pd.read_csv(data_file)
    # read the label csv file
    label_df = pd.read_csv(label_file)
    # prune out the labels without 1 or -1
    label_df = label_df[label_df.rater_1 != 0]
    # replace the label -1 with 0
    label_df = label_df.replace(-1, 0)
    # remove the column on site
    label_df = label_df.drop(columns=['site'])
    # set the column with subject id as index of dataframe
    label_df = label_df.set_index('subject_id')
    # set the column with subject id as index of dataframe
    data_df = data_df.set_index('subject_id')
    # rename label column
    label_df = label_df.rename(columns={'rater_1': 'label'})

    # sort the dataframe by the subject_id as index
    label_df.sort_index(inplace=True)
    data_df.sort_index(inplace=True)

    # slice the training data from the entire data corpus
    data_df = data_df[data_df.index.isin(label_df.index)]
    # ensure the bijective mapping between labels and data in case of missing data
    label_df = label_df[label_df.index.isin(data_df.index)]

    # concatenate the data with the label
    data_df_cat = pd.concat([data_df, label_df], axis=1)

    # debug
    # print(data_df_cat.shape)
    # write dataframe to csv without index and header
    data_df_cat.to_csv(out_file, sep=',', header=False, index=False)

    #debug
    # print(data_df_cat.head())

def test_loading():
    data_file = './abide_train.csv'
    raw_data = np.genfromtxt(data_file, delimiter=',')
    data = raw_data[:,:-1]
    label = raw_data[:,-1]
    print(data.shape)
    print(label.shape)


if __name__=='__main__':
    # label_files = ['./train_2.csv', './val_2.csv']
    # out_files = ['./abide_train.csv', './abide_val.csv']
    # create_train_val_files(label_files, out_files)
    # create_test_file()
    test_loading()





