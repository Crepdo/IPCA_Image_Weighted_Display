from sklearn.decomposition import IncrementalPCA
import numpy as np
import cv2

folder_path = "./output_img"
input=[]
INPUT_NUM = 847
for i in range(1, INPUT_NUM):
    if i%10 == 0: print("loading",i,"th image")
    # if i == 60: continue #special case, should be skipped
    image_path = folder_path+f"/{i}neutral.jpg"
    img = cv2.imread(image_path)
    # 2048x2048 resized to 1024x1024
    img = cv2.resize(img,None, fx=650/2048, fy=650/2048, interpolation=cv2.INTER_AREA)
    input.append(img.reshape(-1))
print("Loaded all",i,"images")
# change into numpy matrixs
input = np.stack(input,axis=0)
# trans to 0-1 format float64
input = (input.astype(np.float16))

# IPCA
COM_NUM = 200
CHUNK_SIZE = 212
pca=IncrementalPCA(n_components = COM_NUM, copy=False,batch_size=212)
print("finished IPCA model set")

element_num = input.shape[0] # how many elements(rows) we have in the dataset
chunk_size = CHUNK_SIZE # how many elements we feed to IPCA at a time, the divisor of n
tail_batch_size = element_num - (element_num//chunk_size)*chunk_size
if (tail_batch_size >0 and tail_batch_size < COM_NUM) or (COM_NUM > chunk_size) :
    print("Invalid batch size setting: Must >= components number !")
    quit()

# fit
for i in range(0, element_num//chunk_size):
    pca.partial_fit(input[i*chunk_size : (i+1)*chunk_size])
    print("finished PCA fit:",i*chunk_size,"to",(i+1)*chunk_size)
if((i+1)*chunk_size < element_num): #tail
    pca.partial_fit(input[(i+1)*chunk_size : element_num])
print("finished PCA fit:",(i+1)*chunk_size,"to",element_num)

result = pca.components_
# result_coff = pca.components_/np.sqrt(pca.explained_variance_.reshape(-1,1))
sv = pca.singular_values_
std_list = []
for i in range(0,COM_NUM):
    std = sv[i]*2/(INPUT_NUM**(1/2))
    result[i] = result[i]*std
    std_list.append(std)
print(std_list)

print("finished components construction")
print(result.shape)
# print("PCA mean:",pca.mean_)

ratio = np.sum(pca.explained_variance_ratio_)*100
print("Components explain %.2f" % ratio,"%","of dataset")

for i in range(0,COM_NUM): result[i] = result[i]/pca.explained_variance_ratio_[i]
print(result)

print("var:")
print(pca.explained_variance_)
print("var_ratio:")
print(pca.explained_variance_ratio_)

result = result.reshape(-1,650,650,3)
print(result.shape)
np.save('reweighted_basis_200.npy', result)
# result shape: #_of_componets * 1024 * 1024 * 3

"""  
# BELOW JUST FOR VERIFY :
saving_path = "./test2"
dst = result
# dst=result/np.linalg.norm(result,axis=(3),keepdims=True)
for j in range(0,COM_NUM):
    reconImage = (dst)[j]
    # reconImage = reconImage.reshape(4096,4096,3)
    reconImage = np.clip(reconImage,0,255)
    reconImage = reconImage.astype(np.uint8)
    cv2.imwrite(os.path.join(saving_path,("p"+str(j)+".png")),reconImage)
print("Saved",j+1,"principle imgs")
"""