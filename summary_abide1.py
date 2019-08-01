"""
Summarize the abide 1 results
"""
import os
import numpy as np


result_file_name = 'abide1_summary.csv'
result_f = open(result_file_name, 'w')
root_dir = "/mnt/Depo/Datasets/ABIDE1/RawDataBIDS/"
png_dir_list = []
for dir_, _, files in os.walk(root_dir):
    for file_name in files:
        if file_name.split('.')[-1] == 'gz':
            rel_dir = os.path.relpath(dir_, root_dir)
            if 'anat' in str(rel_dir):
                rel_file = os.path.join(root_dir, rel_dir, file_name)
                    
                rel_file_dir = os.path.join(root_dir, rel_dir)
                png_dir = os.path.join(rel_file_dir, 'png')
                if not os.path.exists(png_dir):
                    os.mkdir(png_dir)
                rel_file_name = file_name.split('.')[0]
                png_dir_list.append(png_dir)
for itr_png_dir, png_dir in enumerate(png_dir_list):
    png_dir_loc = ''.join(png_dir[:-3])
    sub_id = int(str(png_dir_loc.split('/')[-3].split('-')[-1]))
    site_id = str(png_dir_loc.split('/')[-4])

    qa_score_filepath = png_dir_loc+'qa_ch_1.csv'
    print(qa_score_filepath)
    if os.path.exists(qa_score_filepath):
        ids = []
        scores = []
        with open(qa_score_filepath, 'r') as qa_f:
            lines = qa_f.readlines()
            for line in lines:
                items = line.split(',')
                scores.append(float(items[-1]))
                idx  = items[1].split('.')[0]
                idx = int(str(''.join(idx[-3:])))
                ids.append(idx)
        scores_ = [score for _, score in sorted(zip(ids, scores))]
        scores_pruned = scores_[99:225]
        score_mean = np.mean(scores_pruned)
        score_std = np.std(scores_pruned)
        result_f.write(f'{sub_id},{site_id},{score_mean},{score_std}\n')
        print(f'{sub_id},{site_id},{score_mean},{score_std}')
result_f.close()
