import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Initialize
pred_avg = []
pred_std = []
nb_photon = np.array([1e3, 1e4, 1e5, 1e6, 1e7])


# 1e3 photons
pred_1e3 = np.array([32.72, 33.05, 30.55, 34.39, 31.38])
pred_1e3_avg = np.mean(pred_1e3)
pred_1e3_std = np.std(pred_1e3)

pred_avg.append(pred_1e3_avg)
pred_std.append(pred_1e3_std)

# 1e4 photons
pred_1e4 = np.array([53.81, 51.32, 52.98, 50.33, 50.66])
pred_1e4_avg = np.mean(pred_1e4)
pred_1e4_std = np.std(pred_1e4)

pred_avg.append(pred_1e4_avg)
pred_std.append(pred_1e4_std)

# 1e5 photons
pred_1e5 = np.array([81.42, 82.77, 78.37, 80.06, 80.40])
pred_1e5_avg = np.mean(pred_1e5)
pred_1e5_std = np.std(pred_1e5)

pred_avg.append(pred_1e5_avg)
pred_std.append(pred_1e5_std)

# 1e6 photons
pred_1e6 = np.array([87.21, 84.26, 85.90, 85.24, 79.67])
pred_1e6_avg = np.mean(pred_1e6)
pred_1e6_std = np.std(pred_1e6)

pred_avg.append(pred_1e6_avg)
pred_std.append(pred_1e6_std)

# 1e7 photons
pred_1e7 = np.array([94.61, 92.81, 93.41, 91.62, 88.62])
pred_1e7_avg = np.mean(pred_1e7)
pred_1e7_std = np.std(pred_1e7)

pred_avg.append(pred_1e7_avg)
pred_std.append(pred_1e7_std)

# Linear fit on semilogx
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(nb_photon), pred_avg)
y_fit = intercept + slope * np.log(nb_photon)

plt.figure(1)
plt.errorbar(nb_photon, pred_avg, yerr=pred_std, xerr=None, fmt='', label="line1")
plt.plot(nb_photon, y_fit, 'r--', label="line2")
plt.xlim(1e2, 1e8)
plt.xscale('log')
plt.xlabel('Nombre de photons (échelle logarithme)')
plt.ylabel('Précision %')
plt.gca().legend(('Fit linéaire','Moyenne'))
plt.show()

print("slope: ", slope)
print("intercept: ", intercept)
print(pred_avg)
print(pred_std)