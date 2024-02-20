# coding: utf-8
import pandas as pd
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# model
import models
from models.Swin import mySwin

import argparse


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='/wangbo/Experiments_CIDEr/RainFormer_11_mscoco/save/4e-4/xe_rl/model-best.pth',  # ############################
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str,
                    default='/wangbo/Experiments_CIDEr/RainFormer_11_mscoco/save/4e-4/xe_rl/infos_rainformer_xe-best.pkl',  # #########################
                    help='path to infos to evaluate')

# Basic options
parser.add_argument('--batch_size', type=int, default=50, help='if > 0 then overrule, otherwise load from checkpoint.')  # ##################################
parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
# parser.add_argument('--language_eval', type=int, default=0,
#                     help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? '
#                          'requires coco-caption code from Github.')
parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? '
                         'requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0, help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=0, help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1, help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0, help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=2,  # #######################################################
                    help='used when sample_max = 1, indicates number of beams in beam search. '
                         'Usually 2 or 3 works well. More is not better. Set this to 1 for '
                         'faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                    help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                    help='used for diverse beam search. Usually from 0.2 to 0.8. '
                         'Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). '
                         'Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0, help='If 1, not allowing same word in a row')

# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                    help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='/wangbo/Dataset/MS_COCO/MS_COCO_384',  # #######################################################
                    help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='/shenyiming/wangbo/Region_features_30_data/data/cocobu_fc',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='/shenyiming/wangbo/Region_features_30_data/data/cocobu_att',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='/shenyiming/wangbo/Region_features_30_data/data/cocobu_box',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='/wangbo/Dataset/MS_COCO/cocotalk_label.h5',  # #################################################
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='/wangbo/Dataset/MS_COCO/cocotalk.json',  # #########################################################
                    help='path to the json file containing additional info and vocab. '
                         'empty = fetch from model checkpoint.')
parser.add_argument('--cnn_weight_dir', type=str, default='',
                    help='path to the directory containing the weights of a model trained on imagenet')
parser.add_argument('--split', type=str, default='test',
                    help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                    help='if nonempty then use this file in DataLoaderRaw (see docs there). '
                         'Used only in MSCOCO test evaluation, where we have a specific json file of '
                         'only test set images.')

parser.add_argument('--input_rel_box_dir', type=str, default='/wangbo/Scene_level_or_data/data/cocobu_box_relative',
                    help="this directory contains the bboxes in relative coordinates for "
                         "the corresponding image features in --input_att_dir")

# misc
parser.add_argument('--id', type=str, default='rainformer_xe',  # ################################################
                    help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=0, help='if we need to print out all beam search beams.')  ############### 1 ################
parser.add_argument('--verbose_loss', type=int, default=0, help='if we need to calculate loss.')


opt = parser.parse_args()


def draw_CAM(swin, model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)

    # 获取模型输出的feature/score
    swin.eval()
    model.eval()
    x_0, x_1, x_2, x_3 = my_swin(img)  # ###################### 取x_3可视化 ####################################

    feature_dmse = model.EncoderDecoder.FuseEncoder(x_0, x_1, x_2, x_3)  # ######### 取feature_dmse可视化 ##############

    feature_refine = model(x_0, x_1, x_2, x_3)  # ######### 取feature_refine可视化 ##############

    features = model.features(img)
    output = model.classifier(features)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘


# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()

my_swin = mySwin().cuda()
my_swin = torch.nn.DataParallel(my_swin)
my_swin.eval()

save_path = ""


import seaborn as sns

dual_dimensional_refining_path = "/wangbo/Visualization/dual_dimensional_refining/vis"

final_teacher_item_emb = final_teacher_item_emb.detach().cpu().numpy()
# final_student_colditem_item_emb = final_student_colditem_item_emb.detach().cpu().numpy()
teacher_emb = pd.DataFrame(final_teacher_item_emb)
# student_emb = pd.DataFrame(final_student_colditem_item_emb)
teacher_corr = teacher_emb.corr()
# student_corr = student_emb.corr()
# difference_corr = teacher_corr - student_corr

# %%

plt.subplots(figsize=(10, 10))
sns.heatmap(difference_corr, annot=False, vmax=0.3, vmin=-0.3, square=True, cmap="RdBu_r", center=0)
plt.savefig(os.path.join(dual_dimensional_refining_path, 'ddr_corr_diff.png'))
plt.show()
