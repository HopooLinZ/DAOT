
import os
import torch.nn as nn
import h5py
import torch
from Networks.HR_Net.seg_hrnet import get_seg_model
# from src.CNNModel.Net.VGG16 import SE_VGG
# from Net.VGG16 import SE_VGG
import matplotlib.image as  mpimg
import matplotlib.pyplot as plt
import LoadImage as lim
import numpy as np
import argparse
import cv2
import torchvision.models as models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Get the path of all pictures
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/data/ShanghaiTech/part_B_final/train_data/images_256/',
                        help="""
                        Directory to read images
                        """)
    parser.add_argument('--output',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/save_file/B/T/Feature256/BtrainB.h5' ,
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--model',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/save_file/A2B_retrain3/model_best.pth' ,
                        help="""
                        Directory to cache
                        """)
    args = parser.parse_args()
    return args
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

if __name__ == '__main__':
    args = parse_args()
    features = [] # store eigenvalues
    paths = [] # Storage path
    datapath = args.datapath
    output = args.output
    img_list = get_imlist(datapath) # Get image data

    print('------------------------------------------\n'
          '       feature extraction starts          \n'
          '------------------------------------------')

    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    print("=> loading checkpoint '{}'".format(args.model))
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval() # Indicates for forecasting

    for idx,image_path in enumerate(img_list):
        
        # Deal with one by one
        img = lim.MyLoader(image_path) # Load picture
        img = lim.transform(img) # Image enhancement processing
        img = img.view(1,3,224,224) # To four-dimensional conversion (use DataLoader may be better --- lazier)
        norm_feat = model(img) # Get features, you can change the network
        features.append(norm_feat.detach().cpu().numpy()) # storage characteristics
        paths.append(image_path) # store the corresponding path
        print(f'----------- Path : {image_path} ------------')

    features = np.array(features)
    print(features,'\n',paths)

    print('------------------------------------------\n'
          '   writing feature extractions results    \n'
          '------------------------------------------')
    parent_dir = os.path.dirname(output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    h5f = h5py.File(output,'w') # create h5 file
    h5f.create_dataset('features',data=features)
    h5f.create_dataset('paths',data=np.string_(paths))
    h5f.close()

    print('------------------------------------------\n'
          '           writing successfully           \n'
          '------------------------------------------')



