import os
root_dir = "/mnt/Depo/Datasets/ABIDE1/RawDataBIDS/"
nii_file_list = []
png_file_list = []
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
                png_file_name = png_dir + '/' + rel_file_name + '.png'
                png_file_list.append(png_file_name)

print(nii_file_list[0])
print(png_file_list[0])
