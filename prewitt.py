from numpy.lib.function_base import median
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("crestamento_4.jpg")


image_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)


median_blur = cv2.medianBlur(image_gray,9)


#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(image_gray, -1, kernelx)
img_prewitty = cv2.filter2D(image_gray, -1, kernely)


result = (img_prewittx + img_prewitty)


# cálculo da quantidade de pixels da folha
_, imgBinaryLeaf = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

leaf_area = np.sum(imgBinaryLeaf == 0) #área da folha em pixels
# fim do cálculo da quantidade de pixels da folha


# cálculo da quantidade de pixels de borda
_, imgBinary = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

border_area = np.sum(imgBinary == 255)  #área da borda em pixels

print('Border pixels: ', border_area)
print('Leaf area: ', leaf_area)
# fim do cálculo da quantidade de pixels de borda

#plt.imshow(imgBinaryLeaf, cmap='gray')

cv2.imwrite('crestamento_prewitt_bi.png', imgBinary)



plt.show()
