import numpy as np
import json
import copy
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt

total_samples  = 1               # Number of samples to generate
prefix         = "Sim_Marta"         # A label for the simulations
dictfile       = "parameters_"+prefix+"_"+str(total_samples)+".json" # Output dictionary file
test_run       = False # Don't make dict, but plot samples instead...

# Choose parameters to vary and the prior-range to vary
# We can look at e.g. the EuclidEmulator2 paper to pick the prior range for the emulator
parameters_to_vary = {
  'Omega_cdm':   [0.2,0.34],  # +-25%, euclid has 20%
  'sigma8':      [0.66,0.98], # +-20%, more than Euclid
  'h':           [0.65,0.70]  # Marta test
}
parameters_to_vary_arr = []
for key in parameters_to_vary:
  parameters_to_vary_arr.append(key)

# Set the fiducial cosmology and simulations parameters
run_param_fiducial = {
  'label':        "FiducialCosmology",
  'outputfolder': "../FML/FML/COLASolver/output_Sim",
  'colaexe':      "../FML/FML/COLASolver/nbody",

  # COLA parameters
  'boxsize':    512.0,
  'Npart':      512,
  'Nmesh':      512,
  'Ntimesteps': 30,
  'Seed':       1234567,
  'zini':       20.0,
  'input_spectra_from_lcdm': "false",
  'sigma8_norm': "true",
  
  # Fiducial cosmological parameters - the ones we sample over will be changed below for each sample
  'cosmo_param': {
    'use_physical_parameters': False,
    'cosmology_model': 'LCDM',
    'gravity_model': 'Marta',
    'h':          0.67,
    'Omega_b':    0.049,
    'Omega_cdm':  0.27,
    'Omega_ncdm': 0.001387,
    'Omega_k':    0.0,
    'omega_b':    0.049    * 0.67**2,
    'omega_cdm':  0.27     * 0.67**2,
    'omega_ncdm': 0.001387 * 0.67**2,
    'omega_k':    0.0      * 0.67**2,
    'omega_fld':  0.0      * 0.67**2,
    'w0':         -1.0, 
    'wa':         0.0,
    'Neff':       3.046,
    'k_pivot':    0.05,
    'A_s':        2.1e-9,
    'sigma8':     0.83,
    'n_s':        0.96,
    'T_cmb':      2.7255,
    'log10fofr0': -5.0,
    'gravity_model_marta_mu0':  0.1,
    'largscale_linear': 'false',
    'kmax_hmpc':  20.0,
  },
 
}

#========================================================================
#========================================================================

# Generate all samples
ranges = []
for key in parameters_to_vary:
  ranges.append(parameters_to_vary[key])
ranges = np.array(ranges)
sampling = LHS(xlimits=ranges)
all_samples = sampling(total_samples)

# Generate the dictionaries
simulations = {}
for count, sample in enumerate(all_samples):
  if test_run:
    print("===========================")
    print("New parameter sample:")
  run_param = copy.deepcopy(run_param_fiducial)
  for i, param in enumerate(parameters_to_vary):
    # change the values of the parameter to vary
    run_param["cosmo_param"][param] = sample[i]
    if test_run:
      print("Setting ", param, " to value ", sample[i])
  label = prefix+str(count)
  run_param['label'] = label
  
  simulations[str(count)] = copy.deepcopy(run_param)

if test_run:
  nparam = len(parameters_to_vary_arr)
  for i in range(nparam):
    for j in range(i+1,nparam):
      plt.plot(all_samples[:,i], all_samples[:,j], "o")
      plt.xlabel(parameters_to_vary_arr[i])
      plt.ylabel(parameters_to_vary_arr[j])
      plt.show()
  exit(1)

# Save to file
with open(dictfile, "w") as f:
  data = json.dumps(simulations)
  f.write(data)

