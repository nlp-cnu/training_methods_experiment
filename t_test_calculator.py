

from scipy import stats
import numpy as np

x1 = [0.92827239,0.929190328,0.929643836,0.930430416,0.933335221]
x2 = [0.935811852,0.933763181,0.933101412,0.939853893,0.940298507]
x3 = [0.926558362,0.925869469,0.924238071,0.926873716,0.932758793]
x4 = [0.938332219,0.933220169,0.935424817,0.941341102,0.943157625]
alpha = 0.01

# Paired Two-tailed T-test
# default is two-tailed
t_statistic, p_value = stats.ttest_rel(x1, x2)
print("mean1 = " + str(np.mean(x1)) + ", mean2 = " + str(np.mean(x2)))
print("related p = ", p_value)
print("p < alpha = ", str(p_value < alpha))
if p_value < alpha:
    print("   statistically significantly different")
else:
    print("   not different")

# Unpaired Two-tailed T-test
# default is two-tailed
t_statistic, p_value = stats.ttest_ind(x1, x2)
print("\nmean1 = " + str(np.mean(x1)) + ", mean2 = " + str(np.mean(x2)))
print ("independent p = ", p_value)
print("p < alpha = ", str(p_value < alpha))
if p_value < alpha:
    print("   statistically significantly different")
else:
    print("   not different")


# 1 sample t-test (I can use this to compare against other papers?)
# Just, generate a bunch of samples (based on different splits), then compare
# that population to the hypothesized one (my results to the ones in paper)
# Assumes data is normally distributed and independence of samples (which seems reasonable)
t_statistic, p_value = stats.ttest_1samp(x1, np.mean(x2))
print ("\none-sample p = ", p_value)
print("p < alpha = ", str(p_value < alpha))
if p_value < alpha:
    print("   statistically significantly different")
else:
    print("   not different")

