from classy import Class
import subprocess
import numpy as np
import json
import sys
import os


dictfile       = "parameters_Sim_Marta_1.json"

# Load dict-of-dict where dict["i"] gives parameters for sim "i"
data = []
with open(dictfile) as json_file:
  data = json.load(json_file)

class Generate:

  def __init__(self, run_param):
    self.run_param = run_param

    # Label for the run
    self.label = run_param['label']
    self.label_GR = self.label + "_GR"
    
    # Folder to store everything (use the full path here)
    self.outputfolder = run_param['outputfolder']
    if not os.path.isdir(self.outputfolder):
      subprocess.run(["mkdir", self.outputfolder])
    self.colaexe = run_param['colaexe']
    
    # CLASS parameters (passed to CLASS in [3] below so check that is correct)
    cosmo_param = run_param["cosmo_param"]
    self.h          = cosmo_param['h']           # Hubble H0 / 100km/s/Mpc
    self.use_physical_parameters = cosmo_param['use_physical_parameters']
    self.Omega_b    = cosmo_param['Omega_b']     # Baryon density
    self.Omega_cdm  = cosmo_param['Omega_cdm']   # CDM density
    self.Omega_ncdm = cosmo_param['Omega_ncdm']  # Massive neutrino density
    self.Omega_k    = cosmo_param['Omega_k']     # Curvature density
    self.Omega_fld  = 1.0 - (self.Omega_b+self.Omega_cdm+self.Omega_ncdm+self.Omega_k)
    self.omega_fld  = self.Omega_fld * self.h**2
    self.omega_b    = self.Omega_b * self.h**2
    self.omega_cdm  = self.Omega_cdm * self.h**2
    self.omega_ncdm = self.Omega_ncdm * self.h**2
    self.omega_k    = self.Omega_k * self.h**2

    self.w0_fld     = cosmo_param['w0']          # CPL parametrization
    self.wa_fld     = cosmo_param['wa']          # CPL parametrization
    self.A_s        = cosmo_param['A_s']         # Spectral ampltitude
    self.sigma8     = cosmo_param['sigma8']      
    self.n_s        = cosmo_param['n_s']         # Spectral index
    self.k_pivot    = cosmo_param['k_pivot']     # Pivot scale in 1/Mpc
    self.T_cmb      = cosmo_param['T_cmb']       # CMB temperature
    self.Neff       = cosmo_param['Neff']
    self.N_ncdm     = 1
    self.N_ur       = self.Neff-self.N_ncdm                # Effective number of MASSLESS neutrino species

    try:
      self.kmax_hmpc = cosmo_param['kmax_hmpc']
    except:
      self.kmax_hmpc = 20.0

    # Fixed COLA parameters
    self.boxsize    = run_param['boxsize']
    self.npart      = run_param['Npart']
    self.nmesh      = run_param['Nmesh']
    self.ntimesteps = run_param['Ntimesteps']
    self.seed       = run_param['Seed']
    self.zini       = run_param['zini']
    self.kcola      = "false"
    self.cosmology  = cosmo_param['cosmology_model']
    self.gravity    = cosmo_param['gravity_model']

    if self.gravity == 'Marta':
        self.gravity_model_marta_mu0 = cosmo_param['gravity_model_marta_mu0']  


    # Some other options for cosmologies and gravities
    # and the code will compute growth-factors and scale it accordingly
    # If MG parameters is passed to hi-class and computed exactly then 
    # you have to use "false" below.
    self.input_spectra_from_lcdm = run_param['input_spectra_from_lcdm']
    # if we want to normalize sigma_8 (at redshift 0.0 at sigma8=0.83) or not:
    self.sigma8_norm = run_param['sigma8_norm']

  def generate_data(self, GR):
    self.name = self.label
    if GR:
      self.name = self.label_GR

    # In case of GR use gravity model GR
    gravity = self.gravity
    cosmology = self.cosmology
    if GR:
      cosmology = "LCDM"
      gravity   = "GR"

    print("Doing model ", self.name, " Gravity: ", gravity, " Cosmology: ", cosmology)

    # [1] Make list of redshifts 
    zarr = np.exp(-np.linspace(np.log(1.0/(1.0+self.zini)),np.log(1.0),100))-1.0
    zarr = np.flip(zarr)
    for i in range(len(zarr)):
      zarr[i] = round(zarr[i],3)
    zlist = str(zarr[0])
    for i in range(1,len(zarr)):
      zlist += ","+str(zarr[i])
    
    # [2] Set class parameters and run class
    params = {
        'root': self.outputfolder+'/class_'+self.name+'_',
        #'write_parameters': 'yes',
        'format': 'camb',
        'output': 'tCl,mPk,mTk',
        'l_max_scalars': 2000,
        'P_k_max_1/Mpc': self.kmax_hmpc,
        'h': self.h,
        'w0_fld': self.w0_fld,
        'wa_fld': self.wa_fld,
        'A_s': self.A_s,
        'n_s': self.n_s, 
        'k_pivot': self.k_pivot,
        'N_ur': self.N_ur,
        'N_ncdm': self.N_ncdm,
        'T_cmb': self.T_cmb,
        'z_pk': zlist,
    }
    params['Omega_b'] = self.Omega_b
    params['Omega_cdm'] = self.Omega_cdm
    params['Omega_k'] = self.Omega_k
    params['Omega_ncdm'] = self.Omega_ncdm
    params['Omega_fld'] = self.Omega_fld

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
   # [3] Extract and output transfer information (in CAMB format suitable for COLA)
    colatransferinfofile = self.outputfolder + " " + str(len(zarr))
#    print(zarr)
    for _z in zarr:
      # Transfer
      transfer = cosmo.get_transfer(z=_z, output_format="camb")
      cols = ['k (h/Mpc)', '-T_cdm/k2', '-T_b/k2', '-T_g/k2', '-T_ur/k2', '-T_ncdm/k2', '-T_tot/k2']
      
      # Pofk
      karr = transfer['k (h/Mpc)']
      karr = karr[:-1]
      pofk_total = np.array([cosmo.pk(_k * self.h, _z)*self.h**3 for _k in karr])
      pofk_cb = np.array([cosmo.pk_cb(_k * self.h, _z)*self.h**3 for _k in karr])
      pofkfilename = "class_power_"+self.name+"_z"+str(_z)+".txt"
      np.savetxt(self.outputfolder + "/" + pofkfilename, np.c_[karr, pofk_cb, pofk_total], header="# k (h/Mpc)  P_tot(k)   P_cb(k)  (Mpc/h)^3")
  
      nextra = 13 - len(cols) # We need 13 columns in input so just add some extra zeros
      output = []
      for col in cols:
        output.append(transfer[col])
      for i in range(nextra):
        output.append(transfer[cols[0]]*0.0)
      output = np.array(output)
      filename = "class_transfer_"+self.name+"_z"+str(_z)+".txt"
      np.savetxt(self.outputfolder + "/" + filename, np.c_[output.T], 
                 header="0: k/h   1: CDM      2: baryon   3: photon  4: nu     5: mass_nu  6: total " +
                 "*7: no_nu *8: total_de *9: Weyl *10: v_CDM *11: v_b *12: (v_b-v_c) (* Not present) simlabel = [" + self.name + "]")
      colatransferinfofile += "\n" + filename + " " + str(_z) 


    # [4] Write transferinfo-file needed by COLA
    colatransferinfofilename = self.outputfolder + "/class_transferinfo_"+self.name+".txt"
    with open(colatransferinfofilename, 'w') as f:
        f.write(colatransferinfofile)
    
    # [6] Write the COLA parameterfile
    colafile = "\
------------------------------------------------------------ \n\
-- Simulation parameter file                                 \n\
-- Include other paramfile into this: dofile(\"param.lua\")  \n\
------------------------------------------------------------ \n\
                                                             \n\
-- Don't allow any parameters to take optional values?       \n\
all_parameters_must_be_in_file = true                        \n\
------------------------------------------------------------ \n\
-- Simulation options                                        \n\
------------------------------------------------------------ \n\
-- Label                                                     \n\
simulation_name = \""+self.name+"\"                          \n\
-- Boxsize of simulation in Mpc/h                            \n\
simulation_boxsize = "+str(self.boxsize)+"                   \n\
                                                             \n\
------------------------------------------------------------ \n\
-- COLA                                                      \n\
------------------------------------------------------------ \n\
-- Use the COLA method                                       \n\
simulation_use_cola = true                                   \n\
simulation_use_scaledependent_cola = "+self.kcola+"          \n\
if simulation_use_cola then                                  \n\
  simulation_enforce_LPT_trajectories = false                \n\
end                                                          \n\
------------------------------------------------------------ \n\
-- Choose the cosmology                                      \n\
------------------------------------------------------------ \n\
-- Cosmology: LCDM, w0waCDM, DGP, JBD, ...                   \n\
cosmology_model = \""+self.cosmology+"\"                     \n\
cosmology_OmegaCDM = "+str(self.Omega_cdm)+"                 \n\
cosmology_Omegab = "+str(self.Omega_b)+"                     \n\
cosmology_OmegaMNu = "+str(self.Omega_ncdm)+"                \n\
cosmology_OmegaLambda = "+str(self.Omega_fld)+"              \n\
cosmology_OmegaK = "+str(self.Omega_k)+"                     \n\
cosmology_Neffective = "+str(self.Neff)+"                    \n\
cosmology_TCMB_kelvin = "+str(self.T_cmb)+"                  \n\
cosmology_h = "+str(self.h)+"                                \n\
cosmology_As = "+str(self.A_s)+"                             \n\
cosmology_ns = "+str(self.n_s)+"                             \n\
cosmology_kpivot_mpc = "+str(self.k_pivot)+"                 \n\
                                                             \n\
-- The w0wa parametrization                                  \n\
if cosmology_model == \"w0waCDM\" then                       \n\
  cosmology_w0 = "+str(self.w0_fld)+"                        \n\
  cosmology_wa = "+str(self.wa_fld)+"                        \n\
end                                                          \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Choose the gravity model                                  \n\
------------------------------------------------------------ \n\
-- Gravity model: GR, DGP, f(R), JBD, Geff, ...              \n\
gravity_model = \""+gravity+"\"                              \n\
                                                             \n\
if gravity_model == \"Marta\" then                           \n\
 -- Parameter mu0, for GR = 0.0                              \n\
  gravity_model_marta_mu0=0.1                                \n\
end                                                          \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Particles                                                 \n\
------------------------------------------------------------ \n\
-- Number of CDM+b particles per dimension                   \n\
particle_Npart_1D = "+str(self.npart)+"                      \n\
-- Factor of how many more particles to allocate space       \n\
particle_allocation_factor = 1.5                             \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Output                                                    \n\
------------------------------------------------------------ \n\
-- List of output redshifts                                  \n\
output_redshifts = {0.0}                                     \n\
-- Output particles?                                         \n\
output_particles = false                                     \n\
-- Fileformat: GADGET, FML                                   \n\
output_fileformat = \"GADGET\"                               \n\
-- Output folder                                             \n\
output_folder = \""+self.outputfolder+"\"                    \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Time-stepping                                             \n\
------------------------------------------------------------ \n\
-- Number of steps between the outputs (in output_redshifts) \n\
timestep_nsteps = {"+str(self.ntimesteps)+"}                 \n\
-- The time-stepping method: Quinn, Tassev                   \n\
timestep_method = \"Quinn\"                                  \n\
-- For Tassev: the nLPT parameter                            \n\
timestep_cola_nLPT = -2.5                                    \n\
-- The time-stepping algorithm: KDK                          \n\
timestep_algorithm = \"KDK\"                                 \n\
-- Spacing of the time-steps in 'a': linear, logarithmic, .. \n\
timestep_scalefactor_spacing = \"linear\"                    \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Initial conditions                                        \n\
------------------------------------------------------------ \n\
-- The random seed                                           \n\
ic_random_seed = "+str(self.seed)+"                          \n\
-- The random generator (GSL or MT19937).                    \n\
ic_random_generator = \"GSL\"                                \n\
-- Fix amplitude when generating the gaussian random field   \n\
ic_fix_amplitude = true                                      \n\
-- Mirror the phases (for amplitude-fixed simulations)       \n\
ic_reverse_phases = false                                    \n\
ic_random_field_type = \"gaussian\"                          \n\
-- The grid-size used to generate the IC                     \n\
ic_nmesh = particle_Npart_1D                                 \n\
-- For MG: input LCDM P(k) and use GR to scale back and      \n\
-- ensure same IC as for LCDM                                \n\
ic_use_gravity_model_GR = "+self.input_spectra_from_lcdm+"   \n\
-- The LPT order to use for the IC                           \n\
ic_LPT_order = 2                                             \n\
-- The type of input:                                        \n\
-- powerspectrum    ([k (h/Mph) , P(k) (Mpc/h)^3)])          \n\
-- transferfunction ([k (h/Mph) , T(k)  Mpc^2)]              \n\
-- transferinfofile (a bunch of T(k,z) files from CAMB)      \n\
ic_type_of_input = \"transferinfofile\"                      \n\
-- When running CLASS we can just ask for outputformat CAMB  \n\
ic_type_of_input_fileformat = \"CAMB\"                       \n\
-- Path to the input                                         \n\
ic_input_filename = \""+colatransferinfofilename+"\"         \n\
-- The redshift of the P(k), T(k) we give as input           \n\
ic_input_redshift = 0.0                                      \n\
-- The initial redshift of the simulation                    \n\
ic_initial_redshift = "+str(self.zini)+"                     \n\
-- Normalize wrt sigma8?                                     \n\
-- If ic_use_gravity_model_GR then this is the sigma8 value  \n\
-- in a corresponding GR universe!                           \n\
ic_sigma8_normalization = "+self.sigma8_norm+"               \n\
ic_sigma8_redshift = 0.0                                     \n\
ic_sigma8 = "+str(self.sigma8)+"                             \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Force calculation                                         \n\
------------------------------------------------------------ \n\
-- Grid to use for computing PM forces                       \n\
force_nmesh = "+str(self.nmesh)+"                            \n\
-- Grid to use for computing PM forces                       \n\
force_nmesh = 512                                            \n\
-- Density assignment method: NGP, CIC, TSC, PCS, PQS        \n\
force_density_assignment_method = \"CIC\"                      \n\
-- The kernel to use for D^2 when solving the Poisson equation                    \n\
-- Options: (fiducial = continuous, discrete_2pt, discrete_4pt)                    \n\
force_greens_function_kernel = \"fiducial\"                    \n\
-- The kernel to use for D when computing forces (with fourier)                    \n\
-- Options: (fiducial = continuous, discrete_2pt, discrete_4pt)                    \n\
force_gradient_kernel = \"fiducial\"                    \n\
-- Include the effects of massive neutrinos when computing                    \n\
-- the density field (density of mnu is the linear prediction)                    \n\
-- Requires: transferinfofile above (we need all T(k,z))                    \n\
force_linear_massive_neutrinos = true                    \n\
-- Experimental feature: Use finite difference on the gravitational                     \n\
-- potential to compute forces instead of using Fourier transforms.                    \n\
force_use_finite_difference_force = false                    \n\
force_finite_difference_stencil_order = 4                    \n\
                                                             \n\
------------------------------------------------------------    \n\
-- Lightcone option    \n\
------------------------------------------------------------    \n\
lightcone = false    \n\
if lightcone then    \n\
  -- The origin of the lightcone in units of the boxsize (e.g. 0.5,0.5,0.5 is the center of the box in 3D)    \n\
  plc_pos_observer = {0.0, 0.0, 0.0}    \n\
  -- The boundary region we use around the shell to ensure we get all particles belonging to the lightcone    \n\
  plc_boundary_mpch = 20.0    \n\
  -- The redshift we turn on the lightcone    \n\
  plc_z_init = 1.0    \n\
  -- The redshift when we stop recording the lightcone    \n\
  plc_z_finish = 0.0    \n\
  -- Replicate the box to match the sky coverage we want?    \n\
  -- If not then we need to make sure boxsize is big enough to cover the sky at z_init    \n\
  plc_use_replicas = true    \n\
  -- Number of dimensions where we do replicas in both + and - direction    \n\
  -- The sky fraction is fsky = 1/2^(ndim_rep - NDIM)    \n\
  -- For 3D: if 0 we get an octant and 3 we get the full sky    \n\
  plc_ndim_rep = 3    \n\
  -- Output gadget    \n\
  plc_output_gadgetfile = false    \n\
  -- Output ascii    \n\
  plc_output_asciifile = false    \n\
  -- To save memory output in batches (we only alloc as many particles as we already have to reduce memory consumption)    \n\
  plc_output_in_batches = true    \n\
    \n\
  -- Make delta(z, theta) maps? This is Healpix maps in 3D where we always use the RING scheme for the maps    \n\
  -- For 2D we use output textfiles with the binning    \n\
  plc_make_onion_density_maps = true    \n\
  if plc_make_onion_density_maps then    \n\
    -- Roughly the size of the size of the bins you want in a    \n\
    -- The exact value we use will depend on the time-steps (but not bigger than 2x this value)    \n\
    -- At minimum we make one map per timestep    \n\
    plc_da_maps = 0.025    \n\
    -- Number of pixels (npix = 4*nside^2). The largest lmax we can get from    \n\
    -- the maps is lmax ~ 2nside    \n\
    plc_nside = 512    \n\
    -- Use chunkpix. Only useful for very sparse maps    \n\
    plc_use_chunkpix = false    \n\
    if plc_use_chunkpix then    \n\
      plc_nside_chunks = 256    \n\
    end    \n\
  end    \n\
end    \n\
    \n\
-- On the fly analysis                                       \n\
------------------------------------------------------------ \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Halofinding                                               \n\
------------------------------------------------------------ \n\
fof = true                                                   \n\
fof_nmin_per_halo = 20                                       \n\
fof_linking_length = 0.2                                     \n\
fof_nmesh_max = 0                                            \n\
fof_buffer_length_mpch = 3.0                                 \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Power-spectrum evaluation                                 \n\
------------------------------------------------------------ \n\
pofk = true                                                  \n\
pofk_nmesh = 128                                             \n\
pofk_interlacing = true                                      \n\
pofk_subtract_shotnoise = false                              \n\
pofk_density_assignment_method = \"PCS\"                     \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Power-spectrum multipole evaluation                       \n\
------------------------------------------------------------ \n\
pofk_multipole = false                                       \n\
pofk_multipole_nmesh = 128                                   \n\
pofk_multipole_interlacing = true                            \n\
pofk_multipole_subtract_shotnoise = false                    \n\
pofk_multipole_ellmax = 4                                    \n\
pofk_multipole_density_assignment_method = \"PCS\"           \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Bispectrum evaluation                                     \n\
------------------------------------------------------------ \n\
bispectrum = false                                           \n\
bispectrum_nmesh = 128                                       \n\
bispectrum_nbins = 10                                        \n\
bispectrum_interlacing = true                                \n\
bispectrum_subtract_shotnoise = false                        \n\
bispectrum_density_assignment_method = \"PCS\"               \n\
  """
  
    print("Writing COLA inputfile")
    colainputfile = self.outputfolder + "/cola_input_"+self.name+".lua"
    with open(colainputfile, 'w') as f:
        f.write(colafile)
    
    # [5] Compute HMCode pofk from class (useful to have)
    #cosmo.struct_cleanup()
    #cosmo.empty()
    #cosmo = Class()
    #params["non_linear"] = "hmcode"
    #cosmo.set(params)
    #cosmo.compute()
    #for _z in zarr:
    #  karr = transfer['k (h/Mpc)']
    #  pofk_total = np.array([cosmo.pk(_k * self.h, _z)*self.h**3 for _k in karr])
    #  pofk_cb = np.array([cosmo.pk_cb(_k * self.h, _z)*self.h**3 for _k in karr])
    #  pofkfilename = "class_power_hmcode_"+self.name+"_z"+str(_z)+".txt"
    #  np.savetxt(self.outputfolder + "/" + pofkfilename, np.c_[karr, pofk_cb, pofk_total], header="# k (h/Mpc)  P_tot(k)   P_cb(k)  (Mpc/h)^3")
    
    # [6] Clean up CLASS 
    print("Cleaning CLASS")
    cosmo.struct_cleanup()
    cosmo.empty()


# Generate input for all sims and for both MG and GR
for key in data:
  run_param = data[key]
  g = Generate(run_param)
  g.generate_data(GR=False)
  g.generate_data(GR=True)




