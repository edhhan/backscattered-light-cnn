import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Initialize
pred_avg = []
pred_std = []
resistance = np.array([1e4, 1e5, 1e6, 1e7, 1e8])


# 1e4
pred_1e4 = np.array([32.9, 32, 26.34, 33.0, 34.13])
pred_1e4_avg = np.mean(pred_1e4)
pred_1e4_std = np.std(pred_1e4)

pred_avg.append(pred_1e4_avg)
pred_std.append(pred_1e4_std)

# 1e5
pred_1e5 = np.array([54.5, 51.49, 53.89, 53.29, 50.8])
pred_1e5_avg = np.mean(pred_1e5)
pred_1e5_std = np.std(pred_1e5)

pred_avg.append(pred_1e5_avg)
pred_std.append(pred_1e5_std)

# 1e6
pred_1e6 = np.array([73.86, 65.86, 64.07, 57.48, 71.25])
pred_1e6_avg = np.mean(pred_1e6)
pred_1e6_std = np.std(pred_1e6)

pred_avg.append(pred_1e6_avg)
pred_std.append(pred_1e6_std)

# 1e7
pred_1e7 = np.array([86.22, 79.04, 86.82, 87.01, 86.82])
pred_1e7_avg = np.mean(pred_1e7)
pred_1e7_std = np.std(pred_1e7)

pred_avg.append(pred_1e7_avg)
pred_std.append(pred_1e7_std)

# 1e7 photons
pred_1e8 = np.array([88.02, 91.61, 85.02, 89.82, 91.69])
pred_1e8_avg = np.mean(pred_1e8)
pred_1e8_std = np.std(pred_1e8)

pred_avg.append(pred_1e8_avg)
pred_std.append(pred_1e8_std)

# Linear fit on semilogx
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(resistance), pred_avg)
y_fit = intercept + slope * np.log(resistance)

plt.figure(1)
plt.errorbar(resistance, pred_avg, yerr=pred_std, xerr=None, fmt='', label="line1")
plt.plot(resistance, y_fit, 'r--', label="line2")
plt.xlim(1e3, 1e9)
plt.xscale('log')
plt.xlabel('Résistance (échelle logarithme)')
plt.ylabel('Précision %')
plt.gca().legend(('Fit linéaire','Moyenne'))
plt.show()

print("slope: ", slope)
print("intercept: ", intercept)

print(pred_avg)
print(pred_std)