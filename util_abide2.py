import os
import sys
import subprocess

root_dir = '/home/ashish/Repos/MRI-QA/'
nii_file = '/home/ashish/Repos/MRI-QA/sub-0050551_T1w.nii.gz'
nii_file_name = nii_file.split('/')[-1].split('.')[0]
cmd_str = 'med2image -i ' + nii_file + ' -d ' + root_dir + ' -o ' + nii_file_name+'.png' + ' -s m'

cmd_str_2 = 'med2image ' + '--inputFile ' + nii_file + ' --outputDir ' + root_dir + ' --outputFileStem ' + nii_file_name+'.png' + ' --outputFileType png --sliceToConvert m'

os.system(cmd_str_2)




