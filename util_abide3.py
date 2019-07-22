import os
import sys


root_dir = "/mnt/Depo/Datasets/ABIDE1/RawDataBIDS/"
nii_file_list = []
png_dir_list = []
for dir_, _, files in os.walk(root_dir):
    for file_name in files:
        if file_name.split('.')[-1] == 'gz':
            rel_dir = os.path.relpath(dir_, root_dir)
            if 'anat' in str(rel_dir):
                rel_file = os.path.join(root_dir, rel_dir, file_name)
                nii_file_list.append(rel_file)
                rel_file_dir = os.path.join(root_dir, rel_dir)
                png_dir = os.path.join(rel_file_dir, 'png')
                if not os.path.exists(png_dir):
                    os.mkdir(png_dir)
                rel_file_name = file_name.split('.')[0]
                png_dir_list.append(png_dir)


for i, nii_file_path in enumerate(nii_file_list):
    nii_file_name = nii_file_path.split('/')[-1].split('.')[0]
    sys_cmd_str = f'med2image --inputFile {nii_file_path} --outputDir {png_dir_list[i]} --outputFileStem {nii_file_name} --outputFileType png --sliceToConvert -1'
    os.system(sys_cmd_str)




