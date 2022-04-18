import numpy as np
import cv2
import time

path = "./principle"
images = []
for i in range(0, 20):
    image_path = path+f"/p{i}.png"
    img = cv2.imread(image_path)
    images.append(np.array(img,dtype=np.int16))
mean_img = cv2.imread(path+f"/mean.png")

# output = np.array(mean_img,dtype=np.int16)
output = np.array(cv2.rotate(mean_img, cv2.ROTATE_180),dtype=np.int16)
print("principle images loaded")

cv2.namedWindow('PCA',cv2.WINDOW_NORMAL)
cv2.resizeWindow('PCA', 4000,4000)
# define a null callback function for Trackbar
def null(x): pass

weights = [0 for i in range(0,20)]
for i in range(0,20):
    cv2.createTrackbar("Prin"+str(i), "PCA", 0, 100, null)
    cv2.setTrackbarPos("Prin"+str(i),'PCA',50)

pre_weights=[0 for i in range(0,20)]
def weighting():
    global output 
    global pre_weights
    display_flag = False
    for i in range(0,20):
        weights[i] = (cv2.getTrackbarPos("Prin"+str(i),'PCA')-50)/50
        if weights[i] != pre_weights[i]:
            print("set principle",i,"'s effect to",str(100*weights[i])+"%")
            output = output + (weights[i]-pre_weights[i])*images[i]
            pre_weights[i] = weights[i]
            display_flag = True

    if display_flag == True:
        print("Freshed   Press Keyboard 'q' to Quit")
        # change into cv2 [0,255] uint8 type array:
        formated_out = np.clip(output,0,255)
        formated_out = formated_out.astype(np.uint8)
        # print(formated_out[0][0])
        cv2.imshow('PCA', formated_out)

cv2.imshow('PCA', mean_img)
print("Press Keyboard 'q' to Quit")
while True:
    weighting()
    # display trackbars and image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows() 