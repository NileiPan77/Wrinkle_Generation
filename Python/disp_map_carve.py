import pyexr
import numpy as np
import cv2
import argparse
def disp_map_carve(disp_map, wrinkle_map):
    # 

    # gaussian blur wrinkle_map
    # wrinkle_blurred = cv2.GaussianBlur(wrinkle_map, (3, 3), 3)

    # divide wrinkle map by 2 at non-white pixels
    # wrinkle_map[wrinkle_map != 255] = wrinkle_map[wrinkle_map != 255] - 127
    # wrinkle_map = wrinkle_map / 255.0
    

    # print('shape of disp_map: ', wrinkle_map.shape)
    
    # # replace the value of disp_map with wrinkle_map where wrinkle_map is not white
    # disp_map[wrinkle_map != 1] -= wrinkle_map[wrinkle_map != 1] * 0.1
    
    return disp_map + wrinkle_map * 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='stack image to 16k')
    parser.add_argument('--input', '-i', type=str, help='input image path')
    parser.add_argument('--input2', '-i2', type=str, help='input image path')
    parser.add_argument('--output', '-o', type=str, help='output image path')

    args = parser.parse_args()
    # load disp_map and wrinkle_map
    disp_map = pyexr.open(args.input).get('B')[:,:,0]
    wrinkle_map = pyexr.open(args.input2).get()[:,:,0]

    # call disp_map_carve
    disp_map = disp_map_carve(disp_map, wrinkle_map)
    
    # stack disp_map to 3 channels
    disp_map = np.stack((disp_map, disp_map, disp_map), axis=2)
    # save disp_map
    pyexr.write(args.output, disp_map)