from chunk import Chunk
from unittest import result
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import cv2
import os

folder_path = "./output_img"
input=[]
for i in range(1, 31):
    if i%10 == 0: print("loading",i,"th image")
    image_path = folder_path+f"/{i}neutral.jpg"
    img = cv2.imread(image_path)
    # 2048x2048 resized to 1024x1024
    img = cv2.resize(img, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    input.append(img.reshape(-1))
print("Loaded all",i,"images")
# change into numpy matrixs
all_image = np.stack(input,axis=0)
# trans to 0-1 format float64
all_image = (all_image.astype(np.float16)-128)/127

### shape: #_of_imag x 1024x1024x3
# PCA, keeps 20 features
COM_NUM=2
pca = PCA(n_components = COM_NUM)
pca.fit(all_image)
result1 = pca.fit_transform(all_image)
saving_path = "./prin_test"

result = pca.components_
sv = pca.singular_values_
print(sv.shape)
for i in range(0,COM_NUM):
    result[i] = result[i]*sv[i]
print(result.shape)
"""
result = pca.components_
sv = pca.singular_values_
print(sv.shape)
for i in range(0,COM_NUM):
    result[i] = result[i]*sv[i]
print(result.shape)
print(result1.shape)
result = np.matmul(result1,result)
"""
print(np.sum(pca.explained_variance_ratio_))
# result += pca.mean_
result=result.reshape(-1,1024,1024,3)
for j in range(0,COM_NUM):
    reconImage = result[j]*127+128
    # reconImage = reconImage.reshape(4096,4096,3)
    # reconImage = np.clip(reconImage,0,255)
    # reconImage = reconImage.astype(np.uint8)
    cv2.imwrite(os.path.join(saving_path,("p"+str(j+8)+".png")),reconImage)
    print("Saved",j+1,"principle imgs")
