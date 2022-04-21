from sklearn.decomposition import PCA
import numpy as np
import cv2
import os

folder_path = "./normal"
input=[]
for i in range(1, 131):
    if i%10 == 0: print("loading",i,"th image")
    if i == 60: continue #special case, should be skipped
    image_path = folder_path+f"/total_matrix_tangent {i}.png"
    img = cv2.imread(image_path)
    input.append(img.reshape(-1))
print("Loaded all",i,"images")
# change into numpy matrix
all_image = np.stack(input,axis=0)
# trans to 0-1 format float64
all_image = (all_image.astype(np.float64))

### shape: #_of_imag x image_pixel_num (50331648 for img_normals case)
# print(all_image)
# print(all_image.shape)

# PCA, keeps 20 features
pca=PCA(n_components=20)
result = pca.fit_transform(all_image)
print("finished PCA")

result = pca.inverse_transform(result)
print("PCA mean:",pca.mean_)

result=result.reshape(-1,4096,4096,3)
# result shape: #_of_componets * 4096 * 4096 * 3
saving_path = "./principle"

mean_img = pca.mean_
mean_img = mean_img.reshape(4096,4096,3)
mean_img = mean_img.astype(np.uint8)
cv2.imwrite(os.path.join(saving_path,("mean.png")),mean_img)

dst = result
# dst=result/np.linalg.norm(result,axis=(3),keepdims=True)
for i in range(20):
    reconImage = (dst)[i]
    # reconImage = reconImage.reshape(4096,4096,3)
    reconImage = np.clip(reconImage,0,255)
    reconImage = reconImage.astype(np.uint8)
    cv2.imwrite(os.path.join(saving_path,("p"+str(i)+".png")),reconImage)
print("Saved",i+1,"principle imgs")