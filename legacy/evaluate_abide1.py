import os
import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import argparse
import six
import imageio
import numbers
     
from nr_model import Model
from fr_model import FRModel


def extract_patches(arr, patch_shape=(32,32,3), extraction_step=32):
    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def quality_estimate(image_path, args):
    FR = True
    if args.REF == "":
        FR = False

    if FR:
        model = FRModel(top=args.top)
    else:
        model = Model(top=args.top)


    cuda.cudnn_enabled = True
    cuda.check_cuda_available()
    xp = cuda.cupy
    serializers.load_hdf5(args.model, model)
    model.to_gpu()


    if FR:
        ref_img = imageio.imread(args.REF)
        patches = extract_patches(ref_img)
        X_ref = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))

    img = imageio.imread(image_path)
    patches = extract_patches(img)
    X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))


    y = []
    weights = []
    batchsize = min(2000, X.shape[0])
    t = xp.zeros((1, 1), np.float32)
    for i in six.moves.range(0, X.shape[0], batchsize):
        X_batch = X[i:i + batchsize]
        X_batch = xp.array(X_batch.astype(np.float32))

        if FR:
            X_ref_batch = X_ref[i:i + batchsize]
            X_ref_batch = xp.array(X_ref_batch.astype(np.float32))
            model.forward(X_batch, X_ref_batch, t, False, n_patches_per_image=X_batch.shape[0])
        else:
            model.forward(X_batch, t, False, X_batch.shape[0])

        y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
        weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))

    y = np.concatenate(y)
    weights = np.concatenate(weights)
    qa = np.sum(y*weights)/np.sum(weights)
    return qa


def evaluate_slices(args):
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
    for itr_png_dir, png_dir in enumerate(png_dir_list):
        png_dir_loc = ''.join(png_dir[:-3])
        qa_score_filepath = png_dir_loc+'qa_ch_1.csv'
        if not os.path.exists(qa_score_filepath):
            with open(qa_score_filepath, 'w') as qa_f:
                png_list = os.listdir(png_dir)
                for itr_png_file, png_file_name in enumerate(png_list):
                    png_file_path = png_dir+'/'+png_file_name
                    qa_score = quality_estimate(png_file_path, args)
                    qa_f.write(f'{itr_png_file},{png_file_path},{qa_score}\n')
                    print(f'{itr_png_file},{png_file_path},{qa_score}')
        else:
            with open(qa_score_filepath, 'r') as qa_f:
                lines = qa_f.readlines()
                for line in lines:
                    print(line)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='evaluate_abide1.py')
    parser.add_argument('--reference', '-r', dest='REF', default="", nargs="?", help='path to reference image, if omitted NR IQA is assumed')
    parser.add_argument('--model', '-m', default='./models/nr_live_patchwise.model', help='path to the trained model')
    parser.add_argument('--top', choices=('patchwise', 'weighted'),
                        default='patchwise', help='top layer and loss definition')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID')
    args = parser.parse_args()


    chainer.global_config.train = False
    chainer.global_config.cudnn_deterministic = True

    evaluate_slices(args)



