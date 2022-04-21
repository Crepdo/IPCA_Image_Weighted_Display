import cv2
folder_path = "./wyw256_img"
image_path = folder_path+f"/0000000.png"
img = cv2.imread(image_path)
print(type(img))
print(img.shape)