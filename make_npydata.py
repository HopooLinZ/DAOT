import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''

shanghai_root = '/public/home/qiuyl/DAOT-master/data/ShanghaiTech'
jhu_root = '/public/home/qiuyl/DAOT-master/data/jhu_crowd_v2.0'
qnrf_root = '/public/home/qiuyl/DAOT-master/data/UCF-QNRF_ECCV18'
NWPU_root='/public/home/qiuyl/DAOT-master/data/NWPU_localization'
A2B_retrain_root='/public/home/qiuyl/DAOT-master/data/A2B_retrain'
A2B_fineturn_root='/public/home/qiuyl/DAOT-master/data/A2B_fineturn'

try:
    A2B_fineturn_path = A2B_fineturn_root+'/images/'
    train_list = []
    for filename in os.listdir(A2B_fineturn_path):
        if filename.split('.')[1] == 'jpg':#筛选jpg文件
            train_list.append(A2B_fineturn_path + filename)

    train_list.sort()
    np.save('./npydata/A2B_fineturn.npy', train_list)

    print("generate A2B fineturn image list successfully")
except:
    print("The A2B_fineturn dataset path is wrong. Please check you path.")

try:
    A2Btrain_path = A2B_retrain_root+'/images/'
    train_list = []
    for filename in os.listdir(A2Btrain_path):
        if filename.split('.')[1] == 'jpg':#筛选jpg文件
            train_list.append(A2Btrain_path + filename)

    train_list.sort()
    np.save('./npydata/A2B_retrain.npy', train_list)

    print("generate A2B retrain image list successfully")
except:
    print("The A2B_retrain dataset path is wrong. Please check you path.")



try:

    shanghaiAtrain_path = shanghai_root + '/part_A_final/train_data/images/'
    shanghaiAtest_path = shanghai_root + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':#筛选jpg文件
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully")
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")

try:
    shanghaiBtrain_path = shanghai_root + '/part_B_final/train_data/images/'
    shanghaiBtest_path = shanghai_root + '/part_B_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully")
except:
    print("The ShanghaiB dataset path is wrong. Please check your path.")

try:
    Qnrf_train_path = qnrf_root + '/train_data/images/'
    Qnrf_test_path = qnrf_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(Qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Qnrf_train_path + filename)
    train_list.sort()
    np.save('./npydata/qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(Qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(Qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/qnrf_test.npy', test_list)
    print("Generate QNRF image list successfully")
except:
    print("The QNRF dataset path is wrong. Please check your path.")

try:

    Jhu_train_path = jhu_root + '/train/images_2048/'
    Jhu_val_path = jhu_root + '/val/images_2048/'
    jhu_test_path = jhu_root + '/test/images_2048/'

    train_list = []
    for filename in os.listdir(Jhu_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Jhu_train_path + filename)
    train_list.sort()
    np.save('./npydata/jhu_train.npy', train_list)

    val_list = []
    for filename in os.listdir(Jhu_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(Jhu_val_path + filename)
    val_list.sort()
    np.save('./npydata/jhu_val.npy', val_list)

    test_list = []
    for filename in os.listdir(jhu_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(jhu_test_path + filename)
    test_list.sort()
    np.save('./npydata/jhu_test.npy', test_list)

    print("Generate JHU image list successfully")
except:
    print("The JHU dataset path is wrong. Please check your path.")


try:
    f = open("./data/NWPU_list/train.txt", "r")
    train_list = f.readlines()

    f = open("./data/NWPU_list/val.txt", "r")
    val_list = f.readlines()

    f = open("./data/NWPU_list/test.txt", "r")
    test_list = f.readlines()

    root = '/home/dkliang/projects/synchronous/dataset/NWPU_localization/images_2048/'
    train_img_list = []
    for i in range(len(train_list)):
        fname = train_list[i].split(' ')[0] + '.jpg'
        train_img_list.append(root + fname)
    np.save('./npydata/nwpu_train_2048.npy', train_img_list)

    val_img_list = []
    for i in range(len(val_list)):
        fname = val_list[i].split(' ')[0] + '.jpg'
        val_img_list.append(root + fname)
    np.save('./npydata/nwpu_val_2048.npy', val_img_list)

    test_img_list = []
    root = root.replace('images', 'test_data')
    for i in range(len(test_list)):
        fname = test_list[i].split(' ')[0] + '.jpg'
        fname = fname.split('\n')[0] + fname.split('\n')[1]
        test_img_list.append(root + fname)

    np.save('./npydata/nwpu_test_2048.npy', test_img_list)
    print("Generate NWPU image list successfully")
except:
    print("The NWPU dataset path is wrong. Please check your path.")

try:
    A2B_fineturn_path = A2B_fineturn_root+'/images/'
    train_list = []
    for filename in os.listdir(A2B_fineturn_path):
        if filename.split('.')[1] == 'jpg':#筛选jpg文件
            train_list.append(A2B_fineturn_path + filename)

    train_list.sort()
    np.save('./npydata/A2B_fineturn.npy', train_list)

    print("generate A2B fineturn image list successfully")
except:
    print("The A2B_fineturn dataset path is wrong. Please check you path.")

try:
    A2Btrain_path = A2B_retrain_root+'/images/'
    train_list = []
    for filename in os.listdir(A2Btrain_path):
        if filename.split('.')[1] == 'jpg':#筛选jpg文件
            train_list.append(A2Btrain_path + filename)

    train_list.sort()
    np.save('./npydata/A2B_retrain.npy', train_list)

    print("generate A2B retrain image list successfully")
except:
    print("The A2B_retrain dataset path is wrong. Please check you path.")