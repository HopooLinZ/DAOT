import numpy as np
import os
import ot
import h5py
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import torch
import torch.nn.functional as F
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/save_file/B/T/Feature256/BtrainB.h5',
                        help="""
                        Directory to read images
                        """)
    parser.add_argument('--gallery',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/save_file/A/S/Feature256/AtrainA.h5' ,
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--cindex',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/save_file2/outC/A2B_256.h5' ,
                        help="""
                        Directory to save ssim matrix
                        """)
    parser.add_argument('--index',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/save_file2/out/A2B_256.h5' ,
                        help="""
                        Directory to save mapping
                        """)
    parser.add_argument('--save_path',
                        type=str,
                        default='/public/home/qiuyl/DAOT-master/data2/A2B_finetune' ,
                        help="""
                        Directory to cache
                        """)

    args = parser.parse_args()
    return args

#计算两张图片的SSIM结构相似性
def ssim(img1, img2, window_size=256, size_average=True):
    img1 = img1.cuda()
    img2 = img2.cuda()
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2, True)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2, True)
    sigma1 = F.avg_pool2d(img1 ** 2, window_size, 1, window_size//2, True) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, window_size, 1, window_size//2, True) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2, True) - mu1 * mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    if size_average:
        return 1- ssim_map.mean()
    else:
        return 1- ssim_map.mean(1).mean(1).mean(1)

def save_img(read_path, save_path):
    image = cv2.imread(read_path)
    cv2.imwrite(save_path, image)
    

if __name__ == '__main__':
    args = parse_args()
    # index_gallery = '/media/whut_zhu/LENOVO_USB_HDD/code/slide_data/A/train_data/Feature256/AtrainMA.h5'
    # index_query = '/media/whut_zhu/LENOVO_USB_HDD/code/slide_data/B/train_data/Feature256/BtrainMA.h5'
    index_gallery = args.gallery
    index_query = args.query
    h5f = h5py.File(index_gallery,'r') # open this file.h5

    f = h5py.File(index_query,'r')
    gallery = h5f['features'] # get features
    query = f['features']
    gallery = np.array(gallery)
    query = np.array(query)
    X = np.reshape(query,(len(query),224,224,1))
    Y = np.reshape(gallery,(len(gallery),224,224,1))
    # X = np.reshape(query,(len(query),224,224,1))

# # 计算源域和目标域之间的距离矩阵
# C = ot.dist(X, Y)
# 计算源域和目标域之间的距离矩阵，采用SSIM作为距离度量
    C = np.zeros((len(X)+1, len(Y)+1))
    for i in range(len(X)):
        C[i, len(Y)] = ssim(torch.tensor(X[i]), torch.tensor(np.ones((224, 224, 1))))
        threshold_X = C[i, len(Y)]
        for j in range(len(Y)):
            C[i,j] = ssim(torch.tensor(X[i]), torch.tensor(Y[j]))
            print(C[i, j])
            print(i,j,len(X)*len(Y))
        # C[i,len(Y)+1] = ssim(torch.tensor(X).resize(len(query),224*224,1), torch.tensor(Y).resize(len(gallery),224*224,1))
        if np.min(C[i,:-1])>threshold_X:
            C[i,:-1] = 0
        print("special:",C[i,len(Y)])
    for j in range(len(Y)):
        # C[len(X) + 1,j] = ssim(torch.tensor(X).resize(len(query),224*224,1), torch.tensor(Y).resize(len(gallery),224*224,1))
        C[len(X), j] = ssim(torch.tensor(np.ones((224,224,1))),torch.tensor(Y[j]))
        threshold_Y = C[len(X), j]
        if np.min(C[:-1,j])>threshold_Y:
            C[:-1,j] = 0
        print("specialj:", C[len(X)],j)
    C[len(X), len(Y)] = ssim(torch.tensor(np.ones((224,224,1))),torch.tensor(np.zeros((224,224,1))))
    threshold = C[len(X), len(Y)]
    # C[len(X) + 1, len(Y) + 1] = ssim(torch.tensor(X).resize(len(query),224*224,1), torch.tensor(Y).resize(len(gallery),224*224,1))
    index = args.cindex
    # index = '/home/whut_zhu/save_file/outC/A2B.h5'
    dirs = os.path.dirname(index)
    if not os.path.exists(dirs): 
        os.makedirs(dirs)
    h = h5py.File(index,'a')
    h.create_dataset('index',data=C)
    h.close()

    Y = np.array(Y)
# C = ssim_distance(X_r,Y_r)
# 使用Sinkhorn算法求解最优传输问题
    gamma = float(0.1)  # 正则化参数
    epsilon = float(1e-2)  # 收敛精度
    reg = float(1e-3)
    a = np.ones((len(X)+1,)) / len(X)+1  # 源域的块概率分布
    b = np.ones((len(Y)+1,)) / len(Y)+1  # 目标域的块概率分布
# a = np.ones((len(X),))  # 源域的块概率分布
# b = np.ones((len(Y),))  # 目标域的块概率分布
    alpha= ot.bregman.sinkhorn(a, b, C,reg)  # 这里使用了ot.sinkhorn()函数
# alpha = ot.sinkhorn2(a,b,C,gamma,reg)
# 获取从X到Y的最佳映射
    if np.max(alpha[:-1,:-1])>0:
        mapping = np.argmax(alpha[:-1,:-1], axis=1)
    else:
        mapping = np.argmax([],dtype=int)

# 输出结果
    print("Mapping from source domain to target domain:")
    index = args.index
    dirs = os.path.dirname(index)
    if not os.path.exists(dirs): 
        os.makedirs(dirs)
    h = h5py.File(index,'a')
    h.create_dataset('index',data=mapping)
    h.close()

#生成源域patches的数据集
    o = h5py.File(index,'r')
    index = o['index'][:]
    gallery = h5f['paths'][:] # get path
    query = f['paths'][:] # get path
    for i in range(len(query)):
        images_path = str(gallery[index[i]])[2:-1]
        print('image_path:',images_path)
        gt_path = str(images_path).replace('images','gt_show')
        h5_path = str(images_path).replace('images','gt').replace('jpg','h5')
        save_path = args.save_path
        filename=os.path.basename(images_path)
       
        save_img_path = save_path + '/images/' + filename
        save_gt_path = save_path + '/gt_show/' + filename
        save_fidt_path = save_path + '/gt_fidt_map/' + filename.replace('.jpg','.h5')

        # print('image:',images_path)
        # print('save:',save_img_path)
        # print('gt:',gt_path)
        # print('gt_save:',save_gt_path)
        # print('h5_path:',h5_path)
        # print('save_fidt_path:',save_fidt_path)

        parent_dir = os.path.dirname(save_img_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        save_img(images_path, save_img_path)
        parent_dir = os.path.dirname(save_gt_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        save_img(gt_path, save_gt_path)
        parent_dir = os.path.dirname(save_fidt_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        shutil.copyfile(h5_path, save_fidt_path)
    print('------------------------------------------\n'
          '       making datasets successfully           \n'
          '------------------------------------------')