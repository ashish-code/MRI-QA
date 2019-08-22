'''
Configuration setting for training and validation

Author: Ashish Gupta
Email: ashishagupta@gmail.com
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/ashish/Repos/MRI-QA/data/ABIDE1/', type=str, help='Root directory path of data')
    parser.add_argument('--train_list', default='/home/ashish/Repos/MRI-QA/data/ABIDE1/train.csv', type=str, help='Path for the list of train images and labels')
    parser.add_argument('--test_list', default='/home/ashish/Repos/MRI-QA/data/ABIDE1/val.csv', type=str, help='path to the list of test images and labels')
    parser.add_argument('--num_classes', default=2, type=int, help="Number of classes for image quality")
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument('--save_intervals',default=100,type=int, help='Interation for saving model')
    parser.add_argument('--n_epochs', default=200, type=int, help='Number of total epochs to run')
    parser.add_argument('--input_D', default=64, type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=64, type=int, help='Input size of height')
    parser.add_argument('--input_W', default=64, type=int, help='Input size of width')
    parser.add_argument('--resume_path',default='',type=str,help='Path for resume model.')
    parser.add_argument('--pretrain_path',default='/home/ashish/Repos/MRI-QA/pretrain/resnet_18.pth',type=str,help='Path for pretrained model.')
    parser.add_argument('--new_layer_names',default=['classifier'],type=list,help='New layer except for backbone')
    parser.add_argument('--no_cuda', default=False, type=bool, help='switch for use of cuda, if True, cuda is not used.')
    parser.add_argument('--gpu_id', default=[0],nargs='+',type=int, help='Gpu id lists')
    parser.add_argument('--model',default='resnet',type=str,help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth',default=10,type=int,help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut',default='B',type=str,help='Shortcut type of resnet (A | B)')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    args = parser.parse_args()
    args.save_folder = "./trails/models/{}_{}".format(args.model, args.model_depth)
    
    return args
