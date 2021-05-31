import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = cv2.imread('../data/images/lions.png')
print(img.shape)
img_ori= cv2.resize(img, (500, 300), interpolation=cv2.INTER_CUBIC)
img = img_ori.reshape(300, 500*3)

U, Sigma, VT = np.linalg.svd(img)

def reconstruct_img(U, Sigma, VT, sval_nums=60):
    img_restruct = (U[:, 0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums, :])
    img_restruct = img_restruct.reshape(300, 500, 3)
    return img_restruct

img_1 = reconstruct_img(U, Sigma, VT, sval_nums=60)
img_2 = reconstruct_img(U, Sigma, VT, sval_nums=120)


fig, ax = plt.subplots(1, 3)

ax[0].imshow(cv2.cvtColor(img_ori.astype(np.uint8), cv2.COLOR_BGR2RGB))
ax[0].set(title="src")
ax[1].imshow(cv2.cvtColor(img_1.astype(np.uint8), cv2.COLOR_BGR2RGB))
ax[1].set(title="nums of sigma = 60")
ax[2].imshow(cv2.cvtColor(img_2.astype(np.uint8), cv2.COLOR_BGR2RGB))
ax[2].set(title="nums of sigma = 120")
plt.show()