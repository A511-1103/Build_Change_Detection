import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import cv2 as cv
import glob
import time
from model.bam import BAM
from model.scseunet import SCSEUNet
from model.unet import UNET
from model.hrnet import HRNET
from model.deeplab import DEEPLAB


def create_model_and_load_weights(scse_weights = 'train/scseunet/scseunet.h5',
                                  bam_weights = 'train/bam/bam.h5',
                                  unet_weights= 'train/unet/unet.h5',
                                  hrnet_weights = 'train/hrnet/hrnet.h5',
                                  deeplab_weights = 'train/deeplab/deeplab.h5'
                                  ):
    '''
    :param scse_weights,bam_weights,unet_weights,hrnet_weights,deeplab_weights:
    五个不同模型所对应的不同权重的路径,权重名称以.h5结尾
    :return: 五个加载完权重的模型
    '''
    '''
    先验知识之孪生神经网络:csdn或者阅读相关论文
    '''
    scseunet = SCSEUNet()
    '''
    关于该模型的具体介绍,见论文：https://arxiv.org/pdf/1803.02579v2.pdf
    关于该模型中的注意力机制,见论文:https://arxiv.org/abs/1709.01507
    '''
    bam = BAM()
    '''
    其中涉及到的网络结构见经典分类网络:
    AlexNet:https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    Inception v1:http://arxiv.org/abs/1409.4842
    VGG:https://arxiv.org/abs/1409.1556
    ResNet:https://arxiv.org/pdf/1512.03385.pdf
    SENet:https://arxiv.org/abs/1709.01507
    Inception v2 and BN:http://arxiv.org/abs/1502.03167
    Inception v3:http://arxiv.org/abs/1512.00567
    Inception ResNet (V4):http://arxiv.org/abs/1602.07261
    模型版本见论文:
    v1:https://arxiv.org/pdf/1412.7062v3.pdf
    v2:https://arxiv.org/pdf/1412.7062v3.pdf
    v3:https://arxiv.org/abs/1706.05587
    v3+:https://arxiv.org/pdf/1802.02611.pdf
    其中涉及到的深度可分离卷积系列见论文:
    xception:https://arxiv.org/abs/1610.02357
    Mobile-Net v1：https://arxiv.org/abs/1704.04861
    Mobile-Net v2：https://arxiv.org/abs/1801.04381
    Mobile-Net v3：https://arxiv.org/abs/1905.02244
    其中设及到的图像注意力机制见论文:
    NON-local:https://arxiv.org/abs/1711.07971
    BAM:https://arxiv.org/pdf/1807.06514.pdf
    CBAM:https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
    SKNet:https://arxiv.org/abs/1903.06586
    '''
    unet = UNET()
    '''
    FCN:分割网络的开山之作:https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
    U-Net:https://arxiv.org/abs/1505.04597
    '''
    hrnet = HRNET()
    '''
    https://arxiv.org/abs/1902.09212
    '''
    deeplab = DEEPLAB()
    '''
    与bam相差不大
    '''

    scseunet.load_weights(scse_weights)
    bam.load_weights(bam_weights)
    unet.load_weights(unet_weights)
    hrnet.load_weights(hrnet_weights)
    deeplab.load_weights(deeplab_weights)

    return scseunet,bam,unet,hrnet,deeplab


def read_images(path1,path2):
    img1 = tf.io.read_file(path1)
    img1 = tf.image.decode_png(img1,channels=3)
    img1 = tf.image.resize(img1,[512*2,512*2])
    images1 = img1
    img1 = tf.cast(img1,tf.float32)/255-1
    img1 = tf.expand_dims(img1,axis=0)

    img2 = tf.io.read_file(path2)
    img2 = tf.image.decode_png(img2,channels=3)
    img2 = tf.image.resize(img2,[512*2,512*2])
    images2 =img2
    img2 = tf.cast(img2,tf.float32)/255-1
    img2 = tf.expand_dims(img2,axis=0)

    return img1, img2, images1, images2


def detect(model,img1,img2):
    result = model([img1,img2],training=False)
    test_output = tf.argmax(result,axis=-1)
    test_output = test_output[...,tf.newaxis]
    test_output = tf.squeeze(test_output)
    return test_output


def hrnet_detect(model,img1,img2):
    image = tf.concat([img1,img2],axis=-1)
    result = model(image,training=False)
    test_output = tf.argmax(result,axis=-1)
    test_output = test_output[...,tf.newaxis]
    test_output = tf.squeeze(test_output)
    return test_output


def fill_and_delete(label):
    gray_label = label.copy()
    res1 = cv.findContours(gray_label, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    contours,idx1 = res1
    for i in range(len(contours)):
        area=cv.contourArea(contours[i])
        if area <= 100:
            # print('正在填充面积小于100的区域')
            cv.drawContours(gray_label,contours,i,0,cv.FILLED)
            continue
    res1 = cv.findContours(gray_label, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    contours1,idx1 = res1
    return gray_label,contours1


def only_plt1(input_img, all_cnt):
    all_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                 [255, 255, 255]]
    for i in range(len(all_cnt)):
        colors = all_color[i % 7]
        cv.drawContours(image=input_img,
                        contours=[all_cnt[i]],
                        contourIdx=-1,
                        color=colors,
                        thickness=3)
    return input_img


def plot_bbox(images,cnt):
    for k in range(len(cnt)):
        rect = cv.minAreaRect(cnt[k])
        #                 print(x,y,w,h,angel)
        poly = np.float32(cv.boxPoints(rect))
        poly = np.int0(poly)

        cv.drawContours(image=images,
                        contours=[poly],
                        contourIdx=-1,
                        color=[255, 0, 0],
                        thickness=3)
    return images


def detect_all(scseunet, bam, unet, hrnet, deeplab,img_a_path,img_b_path,mask_path):
    # os.makedirs('res',exist_ok=True)
    t0 = time.time()
    img1, img2, images1, images2 = read_images(img_a_path, img_b_path)
    mask = cv.imread(mask_path)

    t1 = time.time()
    res1 = detect(scseunet, img1, img2)
    t2 = time.time()
    s1 = t2-t1

    t1 = time.time()
    res2 = detect(bam, img1, img2)
    t2 = time.time()
    s2 = t2-t1

    t1 = time.time()
    res3 = detect(unet, img1, img2)
    t2 = time.time()
    s3 = t2-t1

    t1 = time.time()
    res4 = hrnet_detect(hrnet, img1, img2)
    t2 = time.time()
    s4= t2-t1

    t1 = time.time()
    res5 = detect(deeplab, img1, img2)
    t2 = time.time()
    s5 = t2-t1

    result = res1 + res2 + res3 + res4 + res5
    # result = res2 + res3 + res5

    t1 = time.time()
    result = np.where(result >= 3, 255, 0)
    label = np.array(result, np.uint8)
    label, cnt = fill_and_delete(label)
    images2 = np.array(images2, np.uint8)
    plot_images = only_plt1(images2.copy(), cnt)
    t2 = time.time()
    s6 = t2 -t1

    bbox_images = plot_bbox(images2.copy(),cnt)

    save_images = np.zeros((1024, 1024*6, 3))
    save_images[:, :1024, :] = images1
    save_images[:, 1024:2048:, :] = images2
    save_images[:, 2048:1024 * 3, :] = plot_images

    label = cv.cvtColor(label, cv.COLOR_GRAY2BGR)
    save_images[:, 1024 * 3:1024*4, :] = bbox_images

    save_images[:, 1024*4:1024*5, :] = label
    save_images[:, 1024 * 5:, :] = mask
    print(s1, s2, s3, s4, s5, s6, time.time() - t0)

    cv.imwrite('res/{}.png'.format(i), save_images)


if __name__ == '__main__':
    img_a_path = glob.glob('D:/LEVIR-CD/train/A/*.png')
    img_b_path = glob.glob('D:/LEVIR-CD/train/B/*.png')
    mask_path = glob.glob('D:/LEVIR-CD/train/label/*.png')

    img_a_path.sort(), img_b_path.sort(), mask_path.sort()

    scseunet, bam, unet, hrnet, deeplab = create_model_and_load_weights()
    for i in range(len(img_b_path)):
        detect_all(scseunet, bam, unet, hrnet, deeplab, img_a_path[i], img_b_path[i], mask_path[i])















