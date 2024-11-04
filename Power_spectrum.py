# %% [markdown]
# We obtain the import files (Power Spectrum) using CAMB

# %%
import numpy as np
import matplotlib.pyplot as plt
import camb
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scienceplots

# %%
plt.style.use(['science', 'bright'])
print(f"Versión de CAMB: {camb.__version__}")

# %% [markdown]
# Fiducial parameters

# %%
wb0 = 0.022445
wm0 = 0.143648
h = 0.67
c = 300.000

Omega_b0 = wb0/ (h**2)
H0 = 100 * h
Omega_m0 = wm0/ (h**2)
ns = 0.96 # Spectral index of the primordial density power spectrum
Omega_DE0 = 0.68

# print(f"Omega_b0h2 = {Omega_b0 * h**2}")
# print(f"Omega_m0h2 = {Omega_m0* h**2}")
# Setting number density of radiation to zero--------------------------------------------------------------------------
# print(f"Omega_c0h2 = {(Omega_m0- Omega_b0)* h**2}")

# %%
# Define redshifts
redshifts = [1.65, 1.4, 1.2, 1.0]
epsilon = 1e-4

# Function to set cosmological parameters and get results
def get_camb_results(ns, ns_modifier, wm0, wb0, h, epsilon, H0, redshifts, omch2_modifier):
    params = camb.CAMBparams()
    params.set_cosmology(H0=100 * h, ombh2=wb0, omch2=omch2_modifier(ns, wm0, wb0, epsilon, h))
    params.set_matter_power(redshifts=redshifts, kmax=2)
    params.InitPower.set_params(ns=ns_modifier(ns, wm0, wb0, epsilon, h))
    return camb.get_results(params)

def omch2_base(ns, wm0, wb0, epsilon, h):
    return (wm0 - wb0)
def omch2_mn_wb0(ns, wm0, wb0, epsilon, h):
    return wm0 - (wb0* (1 + epsilon))
def omch2_pl_wb0(ns, wm0, wb0, epsilon, h):
    return wm0 - (wb0 * (1 - epsilon))
def omch2_mn_wm0(ns, wm0, wb0, epsilon, h):
    return (wm0 * (1 - epsilon)) - wb0
def omch2_pl_wm0(ns, wm0, wb0, epsilon, h):
    return (wm0 * (1 + epsilon)) - wb0

def ns_base(ns, wm0, wb0, epsilon, h):
    return ns
def ns_mn(ns, wm0, wb0, epsilon, h):
    return ns - epsilon
def ns_pl(ns, wm0, wb0, epsilon, h):
    return ns + epsilon

# Get results for different scenarios
results = get_camb_results(ns, ns_base, wm0, wb0, h, epsilon, H0, redshifts, omch2_base)
results_mn_b0 = get_camb_results(ns, ns_base, wm0, wb0, h, epsilon, H0, redshifts, omch2_mn_wb0)
results_pl_b0 = get_camb_results(ns, ns_base, wm0, wb0, h, epsilon, H0, redshifts, omch2_pl_wb0)
results_mn_m0 = get_camb_results(ns, ns_base, wm0, wb0, h, epsilon, H0, redshifts, omch2_mn_wm0)
results_pl_m0 = get_camb_results(ns, ns_base, wm0, wb0, h, epsilon, H0, redshifts, omch2_pl_wm0)

results_ns_mn = get_camb_results(ns, ns_mn, wm0, wb0, h, epsilon, H0, redshifts, omch2_base)
results_ns_pl = get_camb_results(ns, ns_pl, wm0, wb0, h, epsilon, H0, redshifts, omch2_base)

# %% [markdown]
# The linear Power Spectrum (Pk)

# %%
kh, zs, Pk = results.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True)
kh, zs, Pk_mn_b0 = results_mn_b0.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True)
kh, zs, Pk_pl_b0 = results_pl_b0.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True)
kh, zs, Pk_mn_m0 = results_mn_m0.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True)
kh, zs, Pk_pl_m0 = results_pl_m0.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True)

kh, zs, Pk_ns_mn = results_ns_mn.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True)
kh, zs, Pk_ns_pl = results_ns_pl.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True)

# %% [markdown]
# The Non-linear Power Spectrum (Pk_nonlinear)

# %%
kh_nonlinear, zs, Pk_nonlinear = results.get_linear_matter_power_spectrum(hubble_units=True, k_hunit= True, nonlinear=True)

# %%
# fig, ax = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(6, 4), constrained_layout = True, dpi=300)

# for z in range(len(redshifts)):
#     ax.loglog(kh, Pk[z,:], label=f'z={redshifts[z]}')

# ax.legend(loc='upper right')
# ax.set_title('Matter Power Spectrum')
# ax.set_xlabel(r'$k \, [h \, Mpc^{-1}]$')
# ax.set_ylabel(r'$P(k) \, [h^{-3} Mpc^{3}]$')

# plt.show()

# %%
# fig, ax = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(6, 4), constrained_layout = True, dpi=300)

# for z in range(len(redshifts)):
#     ax.loglog(kh, Pk[z,:], color = 'b')
#     ax.loglog(kh_nonlinear, Pk_nonlinear[z,:], '--', color = 'r' )

# ax.legend(['linear', 'non-linear'], loc='upper right')
# ax.set_title('Linear and Nonlinear Matter Power Spectrum')
# ax.set_xlabel(r'$k \, [h \, Mpc^{-1}]$')
# ax.set_ylabel(r'$P(k) \, [h^{-3} Mpc^{3}]$')

#plt.show()

# %%
class Interpolate_PS:
    def __init__(self):
        self.input_names = ['Pdd', 'P_mn_b0', 'P_pl_b0', 'P_mn_m0', 'P_pl_m0', 'P_ns_mn', 'P_ns_pl']
        self.Pk_input_list = [Pk, Pk_mn_b0, Pk_pl_b0, Pk_mn_m0, Pk_pl_m0, Pk_ns_mn, Pk_ns_pl]

    def interpolate_Pk(self, k, z, Pk_input, kh, zs):
        index = np.where(zs == z)[0][0]
        interp_func = interp1d(np.log(kh), np.log(Pk_input[index]), kind='cubic')
        return np.exp(interp_func(np.log(k)))


Interpolation = Interpolate_PS()
input_names = Interpolation.input_names
Pk_input_list = Interpolation.Pk_input_list

for name, Pk_input in zip(input_names, Pk_input_list):
    exec(f"def {name}(k, z): return Interpolation.interpolate_Pk(k, z, Pk_input, kh, zs)")
# Example of usage
#print(f"Pdd(0.1, zs[0]) = {Pdd(0.1, zs[0])}")


# Example graph
# fig, ax = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(6, 4), constrained_layout=True, dpi=300)
# for z in redshifts:
#     ax.loglog(kh, P_mn_b0(kh, z), label=f'z={z}')

# ax.legend(loc='upper right')
# ax.set_title('Derivative with respect to wb of the linear Matter Power Spectrum')
# ax.set_xlabel(r'$k \, [h \, \mathrm{Mpc}^{-1}]$')
# ax.set_ylabel(r'$P(k) \, [h^{-3} \, \mathrm{Mpc}^{3}]$')

# plt.show()


