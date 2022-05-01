import numpy as np
import cv2
import time

path = "./principle_400_100"
images = []

mean_img = cv2.imread(path+f"/mean.png")
output = np.array(mean_img,dtype=np.int16)

for i in range(0, 100):
    image_path = path+f"/p{i}.png"
    img = cv2.imread(image_path)
    # print(type(img))
    delta = np.array(img,dtype=np.int16)-output
    images.append(delta)

print("principle images loaded")

cv2.namedWindow('control_1-50',cv2.WINDOW_NORMAL)
cv2.resizeWindow('control_1-50', 900,900)

cv2.namedWindow('control_51-100',cv2.WINDOW_NORMAL)
cv2.resizeWindow('control_51-100', 900,900)

cv2.namedWindow('PCA result',cv2.WINDOW_NORMAL)
cv2.resizeWindow('PCA result', 2000,2000)
# define a null callback function for Trackbar
def null(x): pass

weights = [0 for i in range(0,100)]
for i in range(0,50):
    cv2.createTrackbar("Prin"+str(i), "control_1-50", 0, 100, null)
    cv2.setTrackbarPos("Prin"+str(i),'control_1-50',50)
for i in range(50,100):
    cv2.createTrackbar("Prin"+str(i), "control_51-100", 0, 100, null)
    cv2.setTrackbarPos("Prin"+str(i),'control_51-100',50)

pre_weights=[0 for i in range(0,100)]
def weighting():
    global output 
    global pre_weights
    display_flag = False
    for i in range(0,100):
        if i<50:
            weights[i] = (cv2.getTrackbarPos("Prin"+str(i),'control_1-50')-50)/50
        else:
            weights[i] = (cv2.getTrackbarPos("Prin"+str(i),'control_51-100')-50)/50
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
        cv2.imshow('PCA result', formated_out)

cv2.imshow('PCA result', mean_img)
print("Press Keyboard 'q' to Quit")
while True:
    weighting()
    # display trackbars and image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows() 