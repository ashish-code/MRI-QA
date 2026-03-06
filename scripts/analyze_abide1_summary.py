import os
import numpy as np 

input_file_name = 'abide1_summary.csv'
output_file_name = 'abide1_summary_sorted.csv'
sub_ids = []
score_mean = []
score_std = []
out_f = open(output_file_name, 'w')
with open(input_file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split(',')
        sub_ids.append(int(items[0]))
        if 'nan' in items[-1]:
            score_mean.append(25.0)
            score_std.append(10.0)
        else:
            score_mean.append(float(items[2]))
            score_std.append(float(items[3]))

for i, sub_id in enumerate(sub_ids):
    out_f.write(f'{sub_id},{score_mean[i]},{score_std[i]}\n')
    print(f'{sub_id},{score_mean[i]},{score_std[i]}')

out_f.close()
