import numpy as np
import matplotlib.pyplot as plt

n = 6

hanm0 = np.zeros(n)
anm0 = np.zeros(n)
igci0 = np.zeros(n)
lingam0 = np.zeros(n)
cam0 = np.zeros(n)
pca_anm0 = np.zeros(n)

# linear
hanm = [0.815, 0.91875, 0.97375, 0.93625, 0.94625, 0.955]
anm = [0.50875, 0.54625, 0.555555556, 0.5125, 0.5725, 0.52375]
igci = [0.53125, 0.53375, 0.539141414, 0.49375, 0.55125, 0.52375]
lingam = [0.65125, 0.7075, 0.76625, 0.7225, 0.71375, 0.79375]
cam = [0.49375, 0.50875, 0.52020202, 0.525, 0.5725, 0.5125]
pca_anm = [0.7475, 0.815, 0.85375, 0.81375, 0.83125, 0.81625]
X = ['50', '100', '250', '500', '1000', '2000']
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
plt.xlabel("N_Samples")
plt.ylabel("Accuracy")

hanm0 += hanm
anm0 += anm
igci0 += igci
lingam0 += lingam
cam0 += cam
pca_anm0 += pca_anm

# sin
hanm = [0.9325, 0.98125, 0.9875, 0.99625, 0.99125, 0.985]
anm = [0.785, 0.78, 0.80125, 0.8325, 0.81375, 0.79875]
igci = [0.7575, 0.775, 0.76, 0.75875, 0.795, 0.7825]
lingam = [0.43875, 0.52625, 0.61125, 0.62375, 0.665, 0.6425]
cam = [0.7725, 0.78125, 0.7925, 0.78625, 0.795, 0.7825]
pca_anm = [0.70875, 0.7575, 0.765, 0.815, 0.7625, 0.75625]

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
plt.xlabel("N_Samples")
plt.ylabel("Accuracy")

hanm0 += hanm
anm0 += anm
igci0 += igci
lingam0 += lingam
cam0 += cam
pca_anm0 += pca_anm

# exp
hanm = [0.95125, 0.91125, 0.96625, 0.99625, 0.9775, 0.98875]
anm = [0.6375, 0.6875, 0.61125, 0.7275, 0.765, 0.70625]
igci = [0.90625, 0.90625, 0.88375, 0.89625, 0.91625, 0.93875]
lingam = [0.42125, 0.66375, 0.855, 0.7975, 0.90375, 0.91625]
cam = [0.79125, 0.7925, 0.7575, 0.77875, 0.8325, 0.77375]
pca_anm = [0.54375, 0.525, 0.5175, 0.58875, 0.5575, 0.55375]

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
plt.xlabel("N_Samples")
plt.ylabel("Accuracy")
plt.show()

hanm0 += hanm
anm0 += anm
igci0 += igci
lingam0 += lingam
cam0 += cam
pca_anm0 += pca_anm

average = [hanm0 / 3, anm0 / 3, igci0 / 3, lingam0 / 3, cam0 / 3, pca_anm0 / 3]
np.savetxt('average_nsample.txt', average, delimiter='\t', fmt='%.4f')
