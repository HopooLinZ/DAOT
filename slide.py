# import the necessary packages
import helpers
import argparse
import time
import cv2
import os
import sys
import h5py
import scipy.io as sio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/data/ShanghaiTech/part_A_final/train_data/gt_show_256/',
                        help="""
                        Directory to save images
                        """)
    parser.add_argument('--path',
                        type=str,
                        default="/public/home/qiuyl/DAOT-master/data/ShanghaiTech/part_A_final/train_data/gt_show/" ,
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--winW',
                        type=int,
                        default=256 ,
                        help="""
                        """)
    parser.add_argument('--winH',
                        type=int,
                        default=256 ,
                        help="""
                        """)
    parser.add_argument('--stepSize',
                        type=int,
                        default=256 ,
                        help="""
                        """)
    parser.add_argument('--data',
                        type=str,
                        default= 'S' ,
                        help="""Source Target
                        """)
    parser.add_argument('--type',
                        type=str,
                        default= 'img' ,
                        help="""img,h5 
                        """)
    args = parser.parse_args()
    return args
# IMAGE_PATH=os.path.join(BASE_DIR,'image','cat.jpg')
# path = ('DB/pictures/part_A_final/train_data/gt_show/IMG_1.jpg')
def slide_window(path,j, dirs):
    IMAGE_PATH = path
    args = parse_args()
# load the image and define the window width and height
    image = cv2.imread(IMAGE_PATH)
    if args.data == 'T':
        image = cv2.resize(image,(1024,768))
    (winW, winH) = (args.winW,args.winH)
    i = 0

# loop over the image pyramid
    for resized in helpers.pyramid2(image, scale=2):
    # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in helpers.sliding_window(resized, stepSize=args.stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW
            d=0
        # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cropImg_clone = resized.copy()
            cv2.rectangle(clone, (x+d, y+d), (x+d + winW, y+d + winH), (0, 255, 0), 2)
            cropImg = cropImg_clone[y+d: (y+d + winH), x+d:(x+d + winW)]  # H,W
            cv2.waitKey(100)
            path_s = dirs +str(j)[:-4]+'_'+str(i+1)+'.jpg' 
            # cv2.imwrite(path[:-4] +'_'+ WinName + '.jpg', cropImg)
            # print(path_s)
            cv2.imwrite(path_s, cropImg)
            i += 1
#         time.sleep(0.025)
def slide_window_h5(path,j,dirs):
    IMAGE_PATH = path
    # print(IMAGE_PATH)
    args = parse_args()
    with h5py.File(path,
                   'r') as h5f:

            fidt_map = h5f['fidt_map'][()]
            kpoint = h5f['kpoint'][()]
            x, y = fidt_map.shape[0:2]
            # fidt_map = fidt_map/2
            x, y = kpoint.shape[0:2]
            (winW, winH) = (args.winW, args.winH)


            i = 0
            # loop over the image pyramid
            for resized_f, resized_k in zip(helpers.pyramid2(fidt_map, scale=2),helpers.pyramid2(kpoint, scale=2)):
                # loop over the sliding window for each layer of the pyramid
                for (x_f, y_f, window_f),(x_k, y_k, window_k) in zip(helpers.sliding_window(resized_f, stepSize=args.stepSize, windowSize=(winW, winH)),helpers.sliding_window(resized_k, stepSize=args.stepSize, windowSize=(winW, winH))):
                    # if the window does not meet our desired window size, ignore it
                    if window_f.shape[0] != winH or window_f.shape[1] != winW:
                        continue

                    # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
                    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
                    # WINDOW

                    # since we do not have a classifier, we'll just draw the window
                    d = 0
                    clone_f = resized_f.copy()
                    clone_k = resized_k.copy()
                    crop_clone_f = resized_f.copy()
                    crop_clone_k = resized_k.copy()
                    cv2.rectangle(clone_f, (x_f+d, y_f+d), (x_f+d + winW, y_f+d + winH), (0, 255, 0), 2)
                    cv2.rectangle(clone_k, (x_k+d, y_k+d), (x_k+d + winW, y_k+d + winH), (0, 255, 0), 2)
                    croph5_f = crop_clone_f[y_f+d: (y_f+d + winH), x_f:(x_f+d + winW)]  # H,W
                    croph5_k = crop_clone_k[y_k+d: (y_k+d + winH), x_k:(x_k+d + winW)]  # H,W
                    # cv2.imshow("Window", clone_f)
                    # cv2.imshow("Window", clone_k)
                    cv2.waitKey(100)
                    path_s = dirs + str(j)[:-3] + '_' + str(i + 1) + '.h5'  # 
                    f = h5py.File(path_s, 'a')
                    f['fidt_map'] = croph5_f
                    f['kpoint'] = croph5_k
                    f.close()
                    i += 1


def get_image_size(path):
    image = cv2.imread(path)
    height=image.shape[0]
    width=image.shape[1]
    return height,width


                    
if __name__ == "__main__":
    args = parse_args()
    dirs = args.save_dir
    if not os.path.exists(dirs):  
        os.makedirs(dirs)
    path = args.path  # 
    if args.type == 'img':
        files = os.listdir(path)
        image = path + files[0]
        for file in files:
            if os.path.splitext(file)[-1] == ".jpg":
                img_path = path + file
                slide_window(img_path, file, dirs)
    if args.type == 'h5':
        files = os.listdir(path)
        for file in files:
            if os.path.splitext(file)[-1] == ".h5":
                img_path = path + file
                slide_window_h5(img_path, file, dirs)
