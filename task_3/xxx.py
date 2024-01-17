import numpy as np
from example import *

S0 = 8.74496
m_n = 0.7168
N = 60

S = 12.2
# epsilon_corrected = z_score_90 * (D_eta_from_image / np.sqrt(N_from_image))
#
# print(epsilon_corrected)

D_eta = calculate_D_eta(m_n)
epsilon = calculate_accuracy_of_p_epsilon(D_eta, N, CI="99%")
abs_accuracy = calculate_abs_accuracy_of_S(epsilon, S=S)
rel_accuracy = calculate_rel_accuracy_of_S0(abs_accuracy, S0=S0)

print(f"epsilon = {epsilon}")
print(f"abs_accuracy = {abs_accuracy}")
print(f"rel_accuracy = {rel_accuracy}")