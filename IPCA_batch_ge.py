from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import cv2
import os

folder_path = "./output_img"
input=[]
for i in range(1, 201):
    if i%10 == 0: print("loading",i,"th image")
    # if i == 60: continue #special case, should be skipped
    image_path = folder_path+f"/{i}neutral.jpg"
    img = cv2.imread(image_path)
    input.append(img.reshape(-1))
print("Loaded all",i,"images")
# change into numpy matrix
all_image = np.stack(input,axis=0)
# trans to 0-1 format float64
all_image = (all_image.astype(np.float16))

### shape: #_of_imag x image_pixel_num (50331648 for img_normals case)
# print(all_image)
# print(all_image.shape)

# PCA, keeps 20 features
COM_NUM=40
pca=IncrementalPCA(n_components = COM_NUM)
print("finished IPCA model set")

saving_path = "./principle847"


element_num = all_image.shape[0] # how many elements(rows) we have in the dataset
chunk_size = 50 # how many elements we feed to IPCA at a time, the divisor of n

for i in range(0, element_num//chunk_size):
    pca.partial_fit(all_image[i*chunk_size : (i+1)*chunk_size])
    print("finished PCA fit:",i*chunk_size,"to",(i+1)*chunk_size)
pca.partial_fit(all_image[(i+1)*chunk_size : element_num]) #tail
print("finished PCA fit:",(i+1)*chunk_size,"to",element_num)

for i in range(0, element_num//chunk_size):
    if i==0:
        result =  pca.transform(all_image[i*chunk_size : (i+1)*chunk_size])
    else:
        tmp = pca.transform(all_image[i*chunk_size : (i+1)*chunk_size])
        result = np.concatenate((result, tmp), axis=0)
    print("finished PCA transform:",i*chunk_size,"to",(i+1)*chunk_size)

tmp = pca.transform(all_image[(i+1)*chunk_size : element_num]) #tail
result = np.concatenate((result, tmp), axis=0)
print("finished PCA transform:",(i+1)*chunk_size,"to",element_num)

result = pca.inverse_transform(result)
print("PCA mean:",pca.mean_)

mean_img = pca.mean_
mean_img = mean_img.reshape(2048,2048,3)
mean_img = mean_img.astype(np.uint8)
cv2.imwrite(os.path.join(saving_path,("mean.png")),mean_img)

result=result.reshape(-1,2048,2048,3)
# result shape: #_of_componets * 2048 * 2048 * 3
dst = result
# dst=result/np.linalg.norm(result,axis=(3),keepdims=True)
for j in range(0,COM_NUM):
    reconImage = (dst)[j]
    # reconImage = reconImage.reshape(4096,4096,3)
    reconImage = np.clip(reconImage,0,255)
    reconImage = reconImage.astype(np.uint8)
    cv2.imwrite(os.path.join(saving_path,("p"+str(j)+".png")),reconImage)
    print("Saved",j+1,"principle imgs")


