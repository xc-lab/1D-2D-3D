import json
import numpy as np
from PIL import Image
import torch
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
# from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from data_utils.utils import *
from models import *
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# from captum.attr import IntegratedGradients
from torchcam.methods import SmoothGradCAMpp
# CAM GradCAM GradCAMpp ISCAM LayerCAM SSCAM ScoreCAM SmoothGradCAMpp XGradCAM
from torchcam.utils import overlay_mask
# from captum.attr import GradientShap
from captum.attr import Occlusion
# from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

image = cv2.imread('./data/testing_data/PD_00007__1_1_01.jpg') # BGR
data = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA) # 128 128 3
img_pil = Image.open('./data/testing_data/PD_00007__1_1_01.jpg')
# plt.imshow(img_pil)
# plt.show()

model = AlexNet()
checkpoint = torch.load('./checkpoints/alexnet/2023_04_14_09_48_00/model_AlexNet_best_X256.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval().to(device)

trans_A = transforms.Compose([
    transforms.ToTensor()
])

input_tensor = trans_A(np.float32(data)).unsqueeze(0).to(device)
pred_logits = model(input_tensor)
pred_softmax = F.softmax(pred_logits, dim=1)
top_n = pred_softmax.topk(2)
print(top_n)

targets = [ClassifierOutputTarget(1)]

from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
# CAM GradCAM GradCAMpp ISCAM LayerCAM SSCAM ScoreCAM SmoothGradCAMpp XGradCAM

target_layers = [model.features[2]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
# cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
# cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
cam_map = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0]


plt.imshow(cam_map)
plt.show()

import torchcam
from torchcam.utils import overlay_mask

result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.6) # alpha越小，原图越淡

plt.imshow(result)
plt.show()