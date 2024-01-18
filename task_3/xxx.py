import numpy as np
from example import *

S0 = 13.0306
m_n = 0.66
N = 50

S = 19.7434
# epsilon_corrected = z_score_90 * (D_eta_from_image / np.sqrt(N_from_image))
#
# print(epsilon_corrected)

D_eta = calculate_D_eta(m_n)
epsilon = calculate_accuracy_of_p_epsilon(D_eta, N, CI="90%")
abs_accuracy = calculate_abs_accuracy_of_S(epsilon, S=S)
rel_accuracy = calculate_rel_accuracy_of_S0(abs_accuracy, S0=S0)

print(f"epsilon = {epsilon}")
print(f"D_eta = {D_eta}")
print(f"abs_accuracy = {abs_accuracy}")
print(f"rel_accuracy = {rel_accuracy}")