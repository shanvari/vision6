import cv2
import numpy as np
import matplotlib.pyplot as plt
#6.1.1
img = cv2.imread("Homeworks/Images/6/Lena.bmp")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

<matplotlib.image.AxesImage at 0x2230c295420>
 
img_HSI = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_H, img_S, img_I = cv2.split(img_HSI)
plt.imshow(img_H, cmap='gray')

<matplotlib.image.AxesImage at 0x2230c2db220>
 
 
plt.imshow(img_S, cmap='gray')

<matplotlib.image.AxesImage at 0x2230c34aec0>
 

plt.imshow(img_I, cmap='gray')

<matplotlib.image.AxesImage at 0x2230d381a80>
 

#6.1.2
Y’UV
Y’UV defines a color space in terms of one luma (Y’) and two chrominance (UV) components. The Y’UV color model is used in the following composite color video standards.
NTSC ( National Television System Committee)
PAL (Phase Alternating Line)
SECAM (Sequential couleur a amemoire, French for “sequential color with memory
def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)

img = cv2.imread('Homeworks/Images/6/Baboon.bmp')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(img_yuv)

lut_u, lut_v = make_lut_u(), make_lut_v()

# Convert back to BGR so we can apply the LUT and stack the images
y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

u_mapped = cv2.LUT(u, lut_u)
v_mapped = cv2.LUT(v, lut_v)

result = np.vstack([img, y, u_mapped, v_mapped])
plt.figure(figsize=(20, 20))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

<matplotlib.image.AxesImage at 0x2230d78be50>
 
#6.2.1
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import math

img = cv2.imread("Homeworks/Images/6/Lena.bmp",0)
plt.imshow(img,cmap='gray')


<matplotlib.image.AxesImage at 0x19fe4a4b7f0>
 
 def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(255**2 / mse)
# 64bit quantization
bins = np.linspace(0, img.max(), 2**6)
digi_image64 = np.digitize(img, bins)
digi_image64 = (np.vectorize(bins.tolist().__getitem__)(digi_image64-1).astype(int))
plt.imshow(digi_image64,cmap='gray')
print(f"mse: {mse(img,digi_image64)}")
print(f"psnr: {compute_psnr(img,digi_image64)}")

mse: 5.921718597412109
psnr: 40.40632595182891
 
# 32bit quantization
bins = np.linspace(0, img.max(), 2**5)
digi_image32 = np.digitize(img, bins)
digi_image32 = (np.vectorize(bins.tolist().__getitem__)(digi_image32-1).astype(int))
plt.imshow(digi_image32,cmap='gray')
print(f"mse: {mse(img,digi_image32)}")
print(f"psnr: {compute_psnr(img,digi_image32)}")
mse: 19.024349212646484
psnr: 35.33770551571983

 
# 16bit quantization
bins = np.linspace(0, img.max(), 2**4)
digi_image16 = np.digitize(img, bins)
digi_image16 = (np.vectorize(bins.tolist().__getitem__)(digi_image16-1).astype(int))
plt.imshow(digi_image16,cmap='gray')
print(f"mse: {mse(img,digi_image16)}")
print(f"psnr: {compute_psnr(img,digi_image16)}")

mse: 77.15763854980469
psnr: 29.257014335041887

 

# 8bit quantization
bins = np.linspace(0, img.max(), 2**3)
digi_image8 = np.digitize(img, bins)
digi_image8 = (np.vectorize(bins.tolist().__getitem__)(digi_image8-1).astype(int))
plt.imshow(digi_image8,cmap='gray')
print(f"mse: {mse(img,digi_image8)}")
print(f"psnr: {compute_psnr(img,digi_image8)}")


mse: 329.1175308227539
psnr: 22.95729344763723

 


#6.2.2
imgRGB = cv2.imread("Homeworks/Images/6/Lena.bmp")
plt.imshow(cv2.cvtColor(imgRGB,cv2.COLOR_BGR2RGB))

<matplotlib.image.AxesImage at 0x19fe41cbf40>
 
# 4
bins = np.linspace(0, imgRGB.max(), 2**3)
digi_image4 = np.digitize(imgRGB, bins)
digi_image4 = (np.vectorize(bins.tolist().__getitem__)(digi_image4-1).astype(int))
digi_image4 = digi_image4.astype(np.uint8)
plt.imshow(cv2.cvtColor(digi_image4, cv2.COLOR_BGR2RGB))


<matplotlib.image.AxesImage at 0x19fe655e530>
 

# 3
bins = np.linspace(0, imgRGB.max(), 2**2)
digi_image3 = np.digitize(imgRGB, bins)
digi_image3 = (np.vectorize(bins.tolist().__getitem__)(digi_image3-1).astype(int))
digi_image3 = digi_image3.astype(np.uint8)
plt.imshow(cv2.cvtColor(digi_image3, cv2.COLOR_BGR2RGB))


<matplotlib.image.AxesImage at 0x19fe52b8760>
 
# 2
bins = np.linspace(0, imgRGB.max(), 2)
digi_image2 = np.digitize(imgRGB, bins)
digi_image2 = (np.vectorize(bins.tolist().__getitem__)(digi_image2-1).astype(int))
digi_image2 = digi_image2.astype(np.uint8)
plt.imshow(cv2.cvtColor(digi_image2, cv2.COLOR_BGR2RGB))
 
#6.2.3
from PIL import Image
image = Image.open('Homeworks/Images/6/Baboon.bmp')
plt.imshow(image)
 
#8
img8 = image.convert(mode='P', palette=1, colors=8)
plt.imshow(img8)

 
#16
img16 = image.convert(mode='P', palette=1, colors=16)
plt.imshow(img16)
 
#32
img32 = image.convert(mode='P', palette=1, colors=32)
plt.imshow(img32)
 

