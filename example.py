"""
Example script to use Eseg model for predictions
NOTE: model can make multiple predictions at once by passing in a matrix
"""

# 1. make sure eseg_model.py is in your path, then load model
from eseg_model import get_eseg_model

model = get_eseg_model()

# 2. to make predictions, pass in an array of the five feature inputs
# Example case: Pd doped into Rh FCC (110) surface

# (CE_host - CE_dopant) / CN_dopant
diff_ce_cn = -0.28

# gordy electronegativity of the host
gordy_eneg_host = 0.0244

# electron affinity (EA)
# (EA_host - EA_dopant)
diff_ea = 0.575

# atomic radius of the dopant
r_dopant = 1.69

# ionization potential (IP) of the dopant
ip_dopant = 8.33686

# actual Eseg (eV):
eseg = -0.239

# create 1 x 5 array of features
x = [[diff_ce_cn, gordy_eneg_host, diff_ea, r_dopant, ip_dopant]]

# get instant prediction using model.predict
predicted_eseg = model.predict(x)[0]

print('-' * 55)
print('Example case: Pd doped into Rh FCC (110) surface'.center(55))
print('-' * 55)
print(f'   Actual Eseg: {eseg} eV')
print(f'Predicted Eseg: {predicted_eseg:.3f} eV')
print(f'Absolute Error: {abs(eseg - predicted_eseg):+.3f} eV')
print('-' * 55)
