import os
import numpy as np 
import matplotlib.pyplot as plt 

input_file_name = 'abide1_summary_sorted.csv'
data = np.genfromtxt(input_file_name, delimiter=',')

data_sorted = data.copy()
data_sorted = data_sorted[data_sorted[:,1].argsort()]




def plot_it(data):
    fig = plt.figure(1, figsize=(15,5))
    plt.subplot(2,1,1)
    plt.plot(list(range(data.shape[0])), data[:,1], color='black')
    plt.xlabel('sorted subject id')
    plt.ylabel('DL score-Avg')
    plt.ylim([0, 80])
    plt.title('AVG-STD QA scores for ABIDE1')
    plt.subplot(2,1,2)
    plt.plot(list(range(data.shape[0])), data[:,2], color='red')
    plt.xlabel('sorted subject id')
    plt.ylabel('DL score-Std')
    plt.ylim([0, 40])
    plt.show()

# plot_it(data_sorted)


rating_file_name = 'y_abide.csv'
rating_data = np.genfromtxt(rating_file_name, delimiter=',', usecols=(0,2,3,4), missing_values=' ', filling_values=np.nan, skip_header=1)

rd_sum = np.nansum(rating_data[:,1:], axis=1)
rd_allplus1 = np.where(rd_sum == 3)
d_allplus1 = data[rd_allplus1]
print(d_allplus1.shape)

d_allplus1 = d_allplus1[d_allplus1[:,1].argsort()]

plot_it(d_allplus1)
