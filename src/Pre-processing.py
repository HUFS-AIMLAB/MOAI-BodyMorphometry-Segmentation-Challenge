import os
import cv2
import random
import numpy as np
import pydicom
import glob
import argparse
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description = 'MOAI 2020 Body Morphology Segmentation Data Pre-Processing')
    parser.add_argument('--base_dir', type = str, default = '../input/body-morphometry-for-sarcopenia', help = 'Challenge Data base path')
    return parser.parse_args()

def Window_image(image, window_center, window_width): #윈도윙 해주기
  img_min = window_center - window_width // 2
  img_max = window_center + window_width // 2
  window_image = image.copy()
  window_image[window_image < img_min] = img_min
  window_image[window_image > img_max] = img_max
  return window_image

def Make_train_data(img_paths, label_paths, img_png_path, gt_png_path):
    for path_img, path_label in tqdm(zip(img_paths, label_paths)):
        name = os.path.split(path_img)[-1][:-4]
        img = pydicom.dcmread(path_img).pixel_array
        img_arr = np.zeros((512,512,3), dtype = np.float16)
        for num in range(100):
            windowed_img = Window_image(img, random.randrange(-250, 250), random.randrange(500,1000))
            windowed_img += abs(windowed_img.min())
            windowed_img = windowed_img.astype(np.float16)
            img_arr = windowed_img / (windowed_img.max())
            
            label = cv2.imread(path_label)
            label_list = np.unique(label)
            label = np.where(label == label_list[1], 1, label)
            label = np.where(label == label_list[2], 2, label)
            label = np.where(label == label_list[3], 3, label)
            label = label.astype(np.uint8)
            
            limg = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype = np.float16)
            for i in range(3):
                limg[:,:,i] = img_arr
            
            ipath = os.path.join(img_png_path, f"{name}{num}.png")
            lpath = os.path.joun(gt_png_path, f"{name}{num}_P.png")
            plt.imsave(ipath, limg)
            cv2.imwrite(lpath, label)
            
def Make_test_data(img_paths, test_png_path):
    for path_img in tqdm(img_paths):
        name = os.path.split(path_img)[-1][:-4]
        img = pydicom.dcmread(path_img).pixel_array
        
        windowed_img = window_image(img, 0, 750)
        windowed_img += abs(windowed_img.min())
        windowed_img = windowed_img.astype(np.float16)
        img_arr = windowed_img / (windowed_img.max())
        
        limg = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype = np.float16)
        for i in range(3):
                limg[:,:,i] = img_arr
        ipath = os.path.join(test_png_path, f"{name}.png")
        plt.imsave(ipath, limg)
    

def main(args):
    train_img = sorted(glob.glob(args.base_dir + '/train/DICOM/*.dcm'))
    train_label = sorted(glob.glob(args.base_dir + '/train/Label/*.png'))
    test_img = sorted(glob.glob(args.base_dir + '/test/DICOM/*.dcm'))
    img_folder_name = 'img(random)_10000_3ch'
    try:
        os.mkdir(f'./{img_folder_name}')
    except:
        pass
    img_png_path = f'./{img_folder_name}'

    gt_folder_name = 'gt(random)_10000_3ch'
    try:
        os.mkdir(f'./{gt_folder_name}')
    except:
        pass
    gt_png_path = f'./{gt_folder_name}'
    
    test_folder_name = 'test(random)_10000'
    try:
        os.mkdir(f'./{test_folder_name}')
    except:
        pass
    test_png_path = f'./{test_folder_name}'
    
    Make_train_data(train_img, train_label, img_png_path, gt_png_path)
    Make_test_data(test_img, test_png_path)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
