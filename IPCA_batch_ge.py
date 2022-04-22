from chunk import Chunk
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import cv2
import os

folder_path = "./output_img"
input=[]
for i in range(1, 847):
    if i%10 == 0: print("loading",i,"th image")
    # if i == 60: continue #special case, should be skipped
    image_path = folder_path+f"/{i}neutral.jpg"
    img = cv2.imread(image_path)
    # 2048x2048 resized to 1024x1024
    img = cv2.resize(img, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    input.append(img.reshape(-1))
print("Loaded all",i,"images")
# change into numpy matrixs
all_image = np.stack(input,axis=0)
# trans to 0-1 format float64
all_image = (all_image.astype(np.float16))

# IPCA
COM_NUM = 200
CHUNK_SIZE = 212
pca=IncrementalPCA(n_components = COM_NUM)
print("finished IPCA model set")

element_num = all_image.shape[0] # how many elements(rows) we have in the dataset
chunk_size = CHUNK_SIZE # how many elements we feed to IPCA at a time, the divisor of n
tail_batch_size = element_num - (element_num//chunk_size)*chunk_size
if (tail_batch_size >0 and tail_batch_size < COM_NUM) or (COM_NUM > chunk_size) :
    print("Invalid batch size setting: Must >= components number !")
    quit()

# fit
for i in range(0, element_num//chunk_size):
    pca.partial_fit(all_image[i*chunk_size : (i+1)*chunk_size])
    print("finished PCA fit:",i*chunk_size,"to",(i+1)*chunk_size)
if((i+1)*chunk_size < element_num): #tail
    pca.partial_fit(all_image[(i+1)*chunk_size : element_num])
    print("finished PCA fit:",(i+1)*chunk_size,"to",element_num)

# transform
"""
for i in range(0, element_num//chunk_size):
    if i==0:
        result =  pca.transform(all_image[i*chunk_size : (i+1)*chunk_size])
    else:
        tmp = pca.transform(all_image[i*chunk_size : (i+1)*chunk_size])
        result = np.concatenate((result, tmp), axis=0)
    print("finished PCA transform:",i*chunk_size,"to",(i+1)*chunk_size)
if((i+1)*chunk_size < element_num):
    tmp = pca.transform(all_image[(i+1)*chunk_size : element_num]) #tail
    result = np.concatenate((result, tmp), axis=0)
    print("finished PCA transform:",(i+1)*chunk_size,"to",element_num)
"""

result = pca.components_
sv = pca.singular_values_
for i in range(0,COM_NUM):
    result[i] = result[i]*sv[i]
print("finished components construction")
print(result.shape)
# print("PCA mean:",pca.mean_)
ratio = np.sum(pca.explained_variance_ratio_)*100
print("Components explain %.2f" % ratio,"%","of dataset") 
saving_path = "./principle"
mean_img = pca.mean_
mean_img = mean_img.reshape(1024,1024,3)
mean_img = mean_img.astype(np.uint8)
cv2.imwrite(os.path.join(saving_path,("mean.png")),mean_img)

result=result.reshape(-1,1024,1024,3)
# result shape: #_of_componets * 1024 * 1024 * 3
dst = result
# dst=result/np.linalg.norm(result,axis=(3),keepdims=True)
for j in range(0,COM_NUM):
    reconImage = (dst)[j]
    # reconImage = reconImage.reshape(4096,4096,3)
    reconImage = np.clip(reconImage,0,255)
    reconImage = reconImage.astype(np.uint8)
    cv2.imwrite(os.path.join(saving_path,("p"+str(j)+".png")),reconImage)
    print("Saved",j+1,"principle imgs")