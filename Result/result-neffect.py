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
hanm = [0.81, 0.866666, 0.9, 0.934, 0.96, 0.965, 0.9725, 0.9825, 0.995416667, 0.99625]
anm = [0.97, 0.973333333, 0.9675, 0.964, 0.965, 0.961428571, 0.9625, 0.959375, 0.959583333, 0.961875]
igci = [0.54, 0.566666667, 0.58, 0.576, 0.57, 0.57, 0.575, 0.57375, 0.57375, 0.5746875]
lingam = [0.555, 0.57, 0.525, 0.528, 0.52666667, 0.52428571, 0.54, 0.533125, 0.56166667, 0.56875]
cam = [0.69, 0.696666667, 0.715, 0.71, 0.7, 0.695714286, 0.71, 0.714375, 0.710416667, 0.7103125]
pca_anm = [0.385, 0.313333333, 0.3075, 0.258, 0.241666667, 0.175714286, 0.18625, 0.181111111, 0.181111111, 0.181111111]
X = ['2', '3', '4', '5', '6', '7', '8', '16', '24', '32']
plt.figure(dpi=300)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title('one-to-many causality')
plt.plot(X, hanm, 'o-')
plt.plot(X, anm, 'x:', linewidth=2.5)
plt.plot(X, igci, 's-.')
plt.plot(X, lingam, 'p--')
plt.plot(X, cam, 'h--', linewidth=0.6)
plt.plot(X, pca_anm, 'v:')
plt.legend(('HANM', 'ANM', 'IGCI', 'LiNGAM', 'CAM', 'PCA-ANM'), loc='lower right')
plt.ylim(-0.1, 1.1)
plt.xlabel("N_Effects")
plt.ylabel("Accuracy")

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.show()
