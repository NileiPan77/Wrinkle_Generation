# repeat arbitraray size image to 16k

import os
import sys
import numpy as np
import cv2
import argparse
import pyexr
def stack_16k(img, size=16384):
    # img: 3d array
    # size: 16k size
    # return: 3d array
    h, w, c = img.shape
    img_16k = np.zeros((size, size, c), dtype=np.float32)

    # repeat img to fill 16k
    for i in range(size//h):
        for j in range(size//w):
            img_16k[i*h:(i+1)*h, j*w:(j+1)*w, :] = img
    # # boundary condition
    # if size%h != 0:
    #     for j in range(size//w):
    #         img_16k[size-h:size, j*w:(j+1)*w, :] = img[:size%h, :, :]
    # if size%w != 0:
    #     for i in range(size//h):
    #         img_16k[i*h:(i+1)*h, size-w:size, :] = img[:, :size%w, :]
    return img_16k

def main():
    parser = argparse.ArgumentParser(description='stack image to 16k')
    parser.add_argument('--input', '-i', type=str, help='input image path')
    parser.add_argument('--output', '-o', type=str, help='output image path')
    parser.add_argument('--size', '-s', type=int, default=16, help='output image size')
    args = parser.parse_args()

    # if a exr file
    if args.input.split('.')[-1] == 'exr':
        img = pyexr.open(args.input).get()
    else:
        img = cv2.imread(args.input)
    img_16k = stack_16k(img, size=args.size*1024)

    # if a exr file
    if args.output.split('.')[-1] == 'exr':
        pyexr.write(args.output, img_16k)
    else:
        cv2.imwrite(args.output, img_16k)

if __name__ == '__main__':
    main()