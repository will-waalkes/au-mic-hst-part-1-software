import os
import warnings
import random
import pickle
from datetime import datetime
from collections import defaultdict
from functools import partial

# Environment configuration
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood, hmc
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect

import jax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import jit, pmap, devices, device_get, lax, local_device_count, random, vmap, block_until_ready
from jax.random import PRNGKey, split
from jax.scipy.optimize import minimize
from jax.scipy.signal import fftconvolve
from jax.scipy.signal import convolve as jax_convolve

# Check devices
print('Available devices:', devices())
print('CPU devices:', devices('cpu'))

# Suppress warnings
warnings.filterwarnings('ignore', message="It appears that you're using a Mac with one of Apple's ARM-based processors")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
import corner

import astropy.units as u
import astropy.constants as c
from astropy.constants import G, m_p
from astropy.table import Table

import shone

import fleck
from fleck.jax import ActiveStar

from scipy.optimize import fmin_powell, curve_fit
from scipy.stats import gaussian_kde

from chromatic import *
from svo_filters import svo
from bt_settl import get_interp_stellar_spectrum
from bt_settl_3d import get_interp_stellar_spectrum_3d

import arviz
import arviz as az

from tqdm.auto import tqdm

def get_bt_settl_native_wavelengths(path=None):
    """
    Get the native wavelength grid from a BT-Settl model file.
    
    Parameters:
    path: path to a BT-Settl model file (uses first file found if None)
    
    Returns:
    wavelengths: native wavelength array in microns
    """
    if path is None:
        path = '../../../../model-spectra/bt-settl/lte*.txt'
        paths = glob(path)
        if len(paths) == 0:
            raise FileNotFoundError(f"No BT-Settl files found at {path}")
        path = paths[0]  # Use first file to get wavelength grid
    
    # Read the first model to get the native wavelength grid
    spectrum = pd.read_csv(
        path,
        comment='#',
        delimiter=r'\s+',
        names=['wavelength', 'flux']
    )
    
    # Convert from Angstrom to microns
    wavelengths = (spectrum['wavelength'].values * u.AA).to(u.um).value
    
    return wavelengths

def get_bin_edges_from_wavelengths(wavelengths):
    """
    Convert wavelength centers to bin edges.
    
    Parameters:
    wavelengths: array of wavelength centers
    
    Returns:
    bin_edges: array of bin edges with length len(wavelengths) + 1
    """
    half_diffs = np.diff(wavelengths) / 2.0
    bin_edges = np.zeros(len(wavelengths) + 1)
    bin_edges[0] = wavelengths[0] - half_diffs[0]
    bin_edges[1:-1] = wavelengths[:-1] + half_diffs
    bin_edges[-1] = wavelengths[-1] + half_diffs[-1]
    return bin_edges

# Step 1: Get native wavelength grid from BT-Settl models
print("Loading native BT-Settl wavelength grid...")
native_wavelengths = get_bt_settl_native_wavelengths()
print(f"Native wavelength range: {native_wavelengths[0]:.3f} - {native_wavelengths[-1]:.3f} µm")
print(f"Number of native wavelength points: {len(native_wavelengths)}")

# Step 2: Create bin edges from native wavelengths
native_bin_edges = get_bin_edges_from_wavelengths(native_wavelengths)

# Step 3: Store full grid for future use
panchromatic_wavelengths_full = native_wavelengths
panchromatic_btsettl_grid_full = get_interp_stellar_spectrum_3d(native_bin_edges)

# Mask to SED range
sed_wl_min = 0.78
sed_wl_max = 1.65
sed_mask = (native_wavelengths >= sed_wl_min) & (native_wavelengths <= sed_wl_max)
masked_wavelengths = native_wavelengths[sed_mask]
masked_bin_edges = get_bin_edges_from_wavelengths(masked_wavelengths)

panchromatic_wavelengths = masked_wavelengths
panchromatic_bin_edges = masked_bin_edges
panchromatic_btsettl_grid = get_interp_stellar_spectrum_3d(panchromatic_bin_edges)

print("\nGrid creation complete!")

species = ['H2O', 'CH4', 'CO2', 'CO','NH3']

visits = {
    'F21': {
        'Grism': 'G141',
        # 'Forward': G_141_for_dict,
        # 'Backward': G_141_back_dict,
        'BJD_times': np.array(pd.read_csv('../../data/F21_bjdtimes.csv')['BJD'][:]) * u.day,
        'time_lower': 2459455.708 * u.day,
        'time_upper': 2459455.738 * u.day,
        'T0 (BJD_TDB)': 2459455.9895 * u.day,
        'exp (s)': 4.9784 * u.s,
        'native resolution': 46.3 * u.angstrom
    },
    'S22': {
        'Grism': 'G102',
        # 'Forward': G_102_for_dict,
        # 'Backward': G_102_back_dict,
        'BJD_times': np.array(pd.read_csv('../../data/S22_bjdtimes.csv')['BJD'][:]) * u.day,
        'time_lower': 2459684.215 * u.day,
        'time_upper': 2459684.243 * u.day,
        'T0 (BJD_TDB)': 2459684.4959 * u.day, # This is 27 planetary orbits after the first transit, + 0.0054 days (the transit arrived 7 minutes late)
        'exp (s)': 9.67632 * u.s,
        'native resolution': 24.6 * u.angstrom
    }
}

systeminfo = {
    'duration (hr)': 3.5 * u.hr,
    'T_orb (d)': 8.463 * u.day,
    'T_rot (d)': 4.86 * u.day,
    'inclination': 89.5,
    'eccentricity': 0.0,
    'longitude_of_periastron': 88.4
}

def read_sensitivity_curve(grism='G141'):
    path = f'../../data/WFC3.IR.{grism}.1st.sens.2.fits'

    response = fits.open(path)

    w = response[1].data['wavelength']/1e4 * u.micron
    s = response[1].data['sensitivity'] * u.cm * u.cm / u.erg
    e = response[1].data['error'] * u.cm * u.cm / u.erg
    
    return w, s, e

@jit
def get_planck_spectrum_jax(T, **kwargs):
    """
    Calculate the surface flux from a thermally emitted surface,
    according to Planck function.

    Parameters
    ----------
    wavelength : Quantity
        The wavelengths at which to calculate,
        with units of wavelength.
    temperature : Quantity
        The temperature of the thermal emitter,
        with units of K.

    Returns
    -------
    surface_flux : Quantity
        The surface flux, evaluated at the wavelengths.
    """

    # define variables as shortcut to the constants we need
    h = 6.62607e-27 # erg s
    k = 1.380649e-16 # erg/K
    c = 2.9979e18 # angstrom/s
    wavelength = panchromatic_wavelengths*1e4

    z = h * c / (wavelength * k * T) # units check out

    # calculate the intensity from the Planck function
    intensity = (2 * h * c**2 / wavelength**5 / (jnp.exp(z) - 1)) # Units are erg/s/A^3

    # calculate the flux assuming isotropic emission
    flux = jnp.pi * intensity * 1e16 # erg / (s * cm^2 * angstrom)

    # return the intensity
    wave_jax = jnp.array(panchromatic_wavelengths)
    flux_jax = jnp.array(flux)

    return wave_jax, flux_jax

@jit
def convolve_spectrum_jax(model_wavelength, model_flux, sigma, kernel_size=5, **kwargs):
    """
    Properly convolve a spectrum with a Gaussian kernel in JAX.
    
    Args:
        model_wavelength: Array of wavelengths (must be evenly spaced!)
        model_flux: Corresponding flux values
        sigma: Standard deviation of Gaussian kernel in wavelength units
        kernel_size: Number of elements in the kernel (odd number recommended)
        
    Returns:
        Convolved flux array
    """
    # Ensure inputs are JAX arrays
    model_wavelength = jnp.asarray(model_wavelength)
    model_flux = jnp.asarray(model_flux)
    
    # Create proper Gaussian kernel
    x = jnp.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
    kernel = jnp.exp(-0.5 * (x/sigma)**2)
    kernel = kernel / jnp.sum(kernel)  # normalize
    
    # Perform convolution
    convolved = jax_convolve(model_flux, kernel, mode='same', method='fft')
    
    return convolved

@jit
def get_BTSettl_spectrum_jax(T, 
                             logg, metallicity, 
                             grid=panchromatic_btsettl_grid, **kwargs):
    """
    Get BT-Settl spectrum for given temperature, logg, and metallicity.
    
    Parameters:
    T : effective temperature (K)
    logg : surface gravity (log10(cm/s^2))
    metallicity : metallicity ([M/H])
    grid : 3D interpolation function from get_interp_stellar_spectrum_3d
    
    Returns:
    wave_jax : wavelength array (micrometers)
    flux_jax : flux array (erg/cm^2/s/Å)
    """
    
    # Pass all three parameters to the 3D grid interpolation
    gridspec = grid(
        jnp.array(T, dtype=jnp.float32),
        jnp.array(logg, dtype=jnp.float32),
        jnp.array(metallicity, dtype=jnp.float32)
    )
    
    # sigma_sb = 5.67e-5  # erg/cm^2/s/K^4
    # nf = (sigma_sb * (T)**4) / (jnp.trapezoid(gridspec, x=jnp.array(panchromatic_wavelengths)*1e4))
    re_normed_flux = gridspec #* nf
    
    wave_jax = jnp.array(panchromatic_wavelengths)
    flux_jax = jnp.array(re_normed_flux)
    
    return wave_jax, flux_jax

F21_SED_err_factor = np.array([11.606,10.016,3.183,2.000,2.718,3.906,6.409,
                               2.0,3.460,3.134,2.821,2.0,2.0,5.550,
                               2.273,2.000,2.000,2.0,5.025,4.385,2.0,
                               2.683,3.160,2.0,3.135,2.128,2.000,2.000,
                               2.0,2.694,3.583,2.0,2.0,3.989,3.019,
                               6.914,17.888,18.138,5.215,8.532,2.183,3.095,
                               2.0,2.0,4.225,5.362,2.000,2.000,2.0,
                               4.008,4.563,7.652,2.144,2.762,3.996,9.622,
                               2.0,2.0,4.172,2.0,2.942,3.088,3.457,
                               2.0,4.303,2.257,4.045,11.133,4.696,2.0,
                               2.0,2.184,2.651,2.481,2.0,3.845,2.0,
                               2.0,8.637,2.695,8.502,2.680,5.431,4.670,
                               2.0,2.0,2.0,2.0,3.382,6.319,2.400,
                               2.0,2.0,2.0,2.878,3.070,10.219,3.012,
                               2.082,7.564,11.554,5.829,2.976,3.876,4.837,
                               7.078,2.0,2.775,2.0,2.000,])

S22_SED_err_factor = ([1.100, 1.100,1.000,1.100,1.100,1.100,
                       1.100,1.100,1.100,1.100,1.187,1.341,1.100,
                       1.100,1.100,1.100,1.012,1.100,1.1,1.283,
                       1.125,1.272,1.362,1.922,1.976,2.007,1.206,
                       1.387,1.357,1.392,1.798,1.202,1.100,1.100,
                       1.1,1.100,1.223,1.100,1.120,1.131,1.100,
                       1.219,1.100,1.100,1.295,1.1,1.1,1.1,
                       1.100,1.100,1.100,1.000,1.1,1.487,1.265,
                       1.1,1.100,1.1,1.364,1.1,1.1,1.1,
                       1.100,1.100,1.069,1.237,1.486,1.222,1.1,
                       1.103,1.100,1.1,1.1,1.268,1.1,1.1,
                       1.1,1.100,1.1,1.223,1.781,1.470,1.1,
                       1.100,1.100,1.185,1.762,2.281,1.247,1.102,
                       1.1,1.482,1.221,1.1,1.118,1.100,1.108,
                       1.134,1.10,1.100,1.100,1.100,1.100,1.100,
                       1.224,1.1,1.1,1.100,1.1,1.1,1.10,
                       1.1,1.1,1.1,1.1,1.1,1.194,1.791,
                       2.815,1.795,1.1,1.321,1.827,1.840,1.222,
                       1.106,1.100,1.1,1.100,1.1,1.1,1.151,
                       1.221,1.100,1.135,1.115,1.100,1.100,])