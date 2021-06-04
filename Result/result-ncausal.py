#! -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

nsample = 500

hanm0 = np.zeros(10)
anm0 = np.zeros(10)
igci0 = np.zeros(10)
lingam0 = np.zeros(10)
cam0 = np.zeros(10)
pca_anm0 = np.zeros(10)

# linear
hanm = [0.995, 0.973333333, 0.985, 0.934, 0.935, 0.915714286, 0.935, 0.896875, 0.848333333, 0.85875]
anm = [0.55, 0.59, 0.580808081, 0.537373737, 0.506666667, 0.501428571, 0.526515152, 0.52, 0.521464646, 0.535625]
igci = [0.53, 0.533333333, 0.497474747, 0.494949495, 0.488333333, 0.485714286, 0.491161616, 0.543125, 0.523569024, 0.529375]
lingam = [0.58, 0.663333333, 0.65, 0.712, 0.786666667, 0.771428571, 0.725, 0.81125, 0.84, 0.8553125]
cam = [0.545, 0.543333333, 0.51010101, 0.523232323, 0.468333333, 0.467142857, 0.477272727, 0.53375, 0.529461279, 0.5346875]
pca_anm = [0.735, 0.72, 0.77, 0.732, 0.783333333, 0.792857143, 0.8175, 0.840625, 0.7975, 0.8228125]
X = ['2', '3', '4', '5', '6', '7', '8', '16', '24', '32']
plt.figure(dpi=300)
plt.title('linear')
plt.plot(X, hanm, 'o-')
plt.plot(X, anm, 'x:', linewidth=2.5)
plt.plot(X, igci, 's-.')
plt.plot(X, lingam, 'p--')
plt.plot(X, cam, 'h--', linewidth=0.6)
plt.plot(X, pca_anm, 'v:')
plt.legend(('HANM', 'ANM', 'IGCI', 'LiNGAM', 'CAM', 'PCA-ANM'), loc='lower right')
plt.ylim(-0.1, 1.1)
plt.xlabel("N_Causes")
plt.ylabel("Accuracy")

hanm0 += hanm
anm0 += anm
igci0 += igci
lingam0 += lingam
cam0 += cam
pca_anm0 += pca_anm

# sin
hanm = [0.995, 1, 1, 0.998, 0.998333333, 0.995714286, 0.99625, 0.949375, 0.903333333, 0.8865625]
anm = [0.823232323, 0.813333333, 0.795, 0.773737374, 0.8, 0.821428571, 0.821428571, 0.803289474, 0.799319728, 0.780625]
igci = [0.792929293, 0.763333333, 0.7525, 0.75959596, 0.741666667, 0.745714286, 0.745714286, 0.773684211, 0.805272109, 0.768125]
lingam = [0.395, 0.493333333, 0.4775, 0.554, 0.59, 0.612857143, 0.62375, 0.738125, 0.770416667, 0.78]
cam = [0.813131313, 0.823333333, 0.7975, 0.785858586, 0.786666667, 0.775714286, 0.775714286, 0.786184211, 0.788265306, 0.7909375]
pca_anm = [0.745, 0.753333333, 0.805, 0.78, 0.748333333, 0.75, 0.815, 0.755625, 0.6975, 0.7365625]

plt.figure(dpi=300)
plt.title('sin')
plt.plot(X, hanm, 'o-')
plt.plot(X, anm, 'x:', linewidth=2.5)
plt.plot(X, igci, 's-.')
plt.plot(X, lingam, 'p--')
plt.plot(X, cam, 'h--', linewidth=0.6)
plt.plot(X, pca_anm, 'v:')
plt.legend(('HANM', 'ANM', 'IGCI', 'LiNGAM', 'CAM', 'PCA-ANM'), loc='lower right')
plt.ylim(-0.1, 1.1)
plt.xlabel("N_Causes")
plt.ylabel("Accuracy")

hanm0 += hanm
anm0 += anm
igci0 += igci
lingam0 += lingam
cam0 += cam
pca_anm0 += pca_anm

# exp
hanm = [0.995, 1, 0.9925, 0.998, 0.998333333, 0.997142857, 0.99625, 0.9525, 0.954166667, 0.934375]
anm = [0.635, 0.673400673, 0.645, 0.664, 0.651666667, 0.661428571, 0.7275, 0.83, 0.825416667, 0.8321875]
igci = [0.82, 0.838383838, 0.8525, 0.848, 0.866666667, 0.867142857, 0.89875, 0.9225, 0.948333333, 0.9534375]
lingam = [0.695, 0.793333333, 0.8275, 0.808, 0.868333333, 0.822857143, 0.7975, 0.80125, 0.833333333, 0.7865625]
cam = [0.77, 0.734006734, 0.725, 0.75, 0.758333333, 0.765714286, 0.77875, 0.8575, 0.875, 0.890625]
pca_anm = [0.695, 0.566666667, 0.55, 0.594, 0.553333333, 0.555714286, 0.58875, 0.523125, 0.475833333, 0.4628125]

plt.figure(dpi=300)
plt.title('exp')
plt.plot(X, hanm, 'o-')
plt.plot(X, anm, 'x:', linewidth=2.5)
plt.plot(X, igci, 's-.')
plt.plot(X, lingam, 'p--')
plt.plot(X, cam, 'h--', linewidth=0.6)
plt.plot(X, pca_anm, 'v:')
plt.legend(('HANM', 'ANM', 'IGCI', 'LiNGAM', 'CAM', 'PCA-ANM'), loc='lower right')
plt.ylim(-0.1, 1.1)
plt.xlabel("N_Causes")
plt.ylabel("Accuracy")
plt.show()

hanm0 += hanm
anm0 += anm
igci0 += igci
lingam0 += lingam
cam0 += cam
pca_anm0 += pca_anm

average = [hanm0 / 3, anm0 / 3, igci0 / 3, lingam0 / 3, cam0 / 3, pca_anm0 / 3]
np.savetxt('average_nscausal.txt', average, delimiter='\t', fmt='%.4f')
