from matplotlib import pyplot as plt
import numpy as np
import cv2

# rawimg = cv2.imread("./cut/142_230_cut.jpg")
# rawimg = cv2.imread("./cut/259_177_cut.jpg")
# rawimg = cv2.imread("./cut/131_296_cut.jpg")
# rawimg = cv2.imread("./cut/119_231_cut.jpg")
# rawimg = cv2.imread("./cut/118_193_cut.jpg")
# rawimg = cv2.imread("./cut/110_247_cut.jpg")
rawimg = cv2.imread("./cut/131_478_cut.jpg")


fig = plt.figure()
fig.add_subplot(2,3,1)
plt.title("raw image")
plt.imshow(rawimg)

fig.add_subplot(2,3,2)
plt.title("grey scale image")

grayscaleimg = cv2.cvtColor(rawimg,cv2.COLOR_BGR2GRAY)
grayscaleimg = cv2.adaptiveThreshold(grayscaleimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                15, 15)

plt.imshow(grayscaleimg,cmap='gray')

# counting non-zero value by row , axis y
row_nz = []
for row in grayscaleimg.tolist():
    row_nz.append(len(row) - row.count(0))
fig.add_subplot(2,3,3)
plt.title("non-zero values on y (by row)")
plt.plot(row_nz)

# counting non-zero value by column, x axis
col_nz = []
for col in grayscaleimg.T.tolist():
    col_nz.append(len(col) - col.count(0))
fig.add_subplot(2,3,4)
plt.title("non-zero values on y (by col)")
plt.plot(col_nz)

##### start split
# first find upper and lower boundary of y (row)
fig.add_subplot(2,3,5)
plt.title("y boudary deleted")
upper_y = 0
for i,x in enumerate(row_nz):
    if x != 0:
        upper_y = i
        break
lower_y = 0
for i,x in enumerate(row_nz[::-1]):
    if x!=0:
        lower_y = len(row_nz) - i
        break
sliced_y_img = grayscaleimg[upper_y:lower_y,:]
plt.imshow(sliced_y_img)

# then we find left and right boundary of every digital (x, on column)
fig.add_subplot(2,3,6)
plt.title("x boudary deleted")
column_boundary_list = []
record = False


end_list = [i for i,x in enumerate(col_nz) if x==0 or x<=col_nz[-1]]
img_list = []
for i,end in enumerate(end_list):
    if i == 0:
        img_list.append(sliced_y_img[:, i:end])
    else:
        img_list.append(sliced_y_img[:,end_list[i-1]:end] )

# for i,x in enumerate(col_nz[:-1]):
#     if (col_nz[i] ==0  and col_nz[i+1] != 0) or col_nz[i] != 0 and col_nz[i+1]== 0:
#         column_boundary_list.append(i+1)
# img_list = []
# xl = [ column_boundary_list[i:i+2] for i in range(0,len(column_boundary_list),2) ]
# for x in xl:
#     img_list.append( sliced_y_img[:,x[0]:x[1]] )
# del invalid image
img_list = [ x for x in img_list if x.shape[1] > 2]
# show image
fig = plt.figure()
for i,img in enumerate(img_list):
    fig.add_subplot(5,5,i+1)
    plt.imshow(img)
    # plt.imsave("./cut/%s.jpg"%i,img)
plt.show()