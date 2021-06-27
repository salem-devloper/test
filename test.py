from glob import glob
from pydicom import dcmread
from tensorflow.keras.models import load_model
import cv2
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

def get_args():

    parser = argparse.ArgumentParser(description = "U-Net for Lung Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path',type=str,default='./test')

    parser.add_argument('--load_model', type=str, default='./test/lung_seg_valiou_088.h5', help='.pth file path to load model')
    return parser.parse_args()

args = get_args()
model=load_model(args.load_model)
os.mkdir('./res')
out = './res'
dirpath= args.path + '/test/*/*.jpeg'
for file in tqdm(sorted(glob(dirpath))):
    print(file)
    img=cv2.imread(file)
    raw_img=img
    width,height=raw_img.shape[1],raw_img.shape[0]
    img=cv2.resize(img,(256,256))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
    mask=model.predict(np.expand_dims(img,axis=0))
    mask[mask<0.5]=0.0
    mask[mask>0.5]=1.0
    img=cv2.imread(file,0)
    img=cv2.resize(img,(256,256)).reshape(256,256)
    masked_img=np.squeeze(img*mask.reshape(256,256))
    masked_img=cv2.resize(masked_img,(width,height)).astype(np.int16)
    plt.imshow(img.astype(np.float32),cmap='gray')
    plt.imshow(mask.reshape(256,256),cmap='Greens',alpha=0.5)
    plt.show()
    Image.fromarray(masked_img).save(os.path.join(out,'mask_'+file))
    

    
