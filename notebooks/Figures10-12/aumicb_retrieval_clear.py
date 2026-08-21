import sys
import astropy.table
from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
#from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import M_jup, R_jup, R_sun, M_earth, R_earth
import numpy as np
import matplotlib.pyplot as plt
import getopt
import pickle
from scipy.stats import kde
from scipy import interpolate
import arviz as az
from platon.plotter import Plotter
import threading
import multiprocessing

def readInDataCSV(fn): 
    infile = open(fn,'r')
    lines = infile.readlines()[1:]
    infile.close()
    nl = len(lines)
    wv_,td_,tderr_ = np.zeros(nl),np.zeros(nl),np.zeros(nl)
    wvbins_ = np.zeros((nl,2))
    for i in range(nl):
        line = lines[i].split(',')
        wv_[i] = float(line[0])/1e6 # m
        wvbins_[i,0] = float(line[1])/1e6 # m
        wvbins_[i,1] = float(line[2])/1e6 # m
        td_[i] = float(line[3])**2 # Rp/Rs -> transit depth 
        tderr_[i] = 2.*float(line[3])*float(line[4]) # Rp/Rs error -> transit depth error
    return wv_,wvbins_,td_,tderr_,nl

# Plot settings
plt.rc('font', family='sans-serif')
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)
plt.rc('axes', linewidth=2)
fontsize = 15
fontname = 'sans-serif'
linewidth = 2
annosize=10
annoweight=100
labelpad=2

# G141
#wv141,wvbins141,td141,tderr141,n141 = readInDataCSV('aumicb_data/F21-transspec-contaminated.csv')
#wv141,wvbins141,td141,tderr141,n141 = readInDataCSV('aumicb_data/F21-aphysical-Dec12.csv')
wv141,wvbins141,td141,tderr141,n141 = readInDataCSV('aumicb_data/F21-contam-blind-Dec28.csv')

#table = astropy.table.Table.read('G141_M3_transmission_spectrum.fits')
#wavelength_center = np.array(table['wavelength_bin_center'])
#blue_edge = np.array(table['wavelength_bin_low'])
#red_edge = np.array(table['wavelength_bin_high'])
#rp_rs = np.array(table['rp_rs'])
#upper_err = np.array(table['rp_rs_upper_err'])
#lower_err = np.array(table['rp_rs_lower_err'])

# G102
#wv102,wvbins102,td102,tderr102,n102 = readInDataCSV('aumicb_data/S22-transspec-contaminated.csv')
#wv102,wvbins102,td102,tderr102,n102 = readInDataCSV('aumicb_data/S22-aphysical-Dec12.csv')
wv102,wvbins102,td102,tderr102,n102 = readInDataCSV('aumicb_data/S22-contam-blind-Dec28.csv')

#print(wv102,wvbins102,td102,tderr102,n102)
#print(wv141,wvbins141,td141,tderr141,n141)

wvbins = np.concatenate((wvbins102,wvbins141))
td = np.concatenate((td102,td141))
tderr = np.concatenate((tderr102,tderr141))

#print(wvbins,td,tderr)


#nw = len(wavelength_center)
#wvbins,td,tderr = np.zeros((nw,2)),np.zeros(nw),np.zeros(nw)
#for i in range(len(wavelength_center)):
#    wvbins[i,0] = blue_edge[i]/1e6 # micron -> m
#    wvbins[i,1] = red_edge[i]/1e6 # micron -> m
#    td[i] = rp_rs[i]**2 # Nothing
#    tderr[i] = 2.*rp_rs[i]*0.5*(upper_err[i]+lower_err[i])

# Read in inputs from command line
#opts, args = getopt.getopt(sys.argv[1:], "h")
#fname = args[0]
#minput = args[1]

#fstr = fname.split('_')[-1].split('.')[0]
#print(fstr)

#if minput == 'mpout':
#    mstr = 'mas24outmp'
#    mval = 6.9
#    merr = 2.9
#elif minput == 'mpin':
#    mstr = 'mas24inmp'
#    mval = 3.7
#    merr = 2.2
#elif minput == 'mpunc2':
#    mstr = 'mas24outmp_2xerr'
#    mval = 6.9
#    merr = 2.9*2.
#elif minput == 'mpunc4':
#    mstr = 'mas24outmp_4xerr'
#    mval = 6.9
#    merr = 2.9*4.
#else: 
#    print('Invalid mass string. Quitting.')
#    quit()

#wv,wvbins,td,tderr = readInData(fname)
#wvbins = data[da]['wvbins']
#td = data[da]['td']
#tderr = data[da]['tderr']
#print(wv,wvbins,td,tderr)

pfolder = 'aumicb_retrieval_results/'

#create a Retriever object
retriever = CombinedRetriever()

#create a FitInfo object and set best guess parameters; assume solar C/O from Asplund et al. 2009
fit_info = retriever.get_default_fit_info(
    Rs=0.8 * R_sun, Mp=13.5 * M_earth,
    Rp=0.045 * 0.82 * R_sun, T=600.0, CO_ratio=0.59, #logMp=np.log10(6.5 * M_earth),
    logZ=1., log_cloudtop_P=np.inf, cloud_frac=1.,
    log_scatt_factor=0, scatt_slope=4, 
    #fit_vmr=True, 
    error_multiple=0.5*np.max(tderr),#[6:]), 
    #T_star=4100.0, offset_start=0,offset_end=n102-1,#-6,
    #T_spot=3500.0, spot_cov_frac=0.5,
    T_star=3891.0, offset_start=0,offset_end=n102-1,#-6,
    T_spot=3020.0, spot_cov_frac=0.5,
    offset_transit=0.0)

#fit_info.add_gases_vmr(["H2O","CH4","CO2","CO","SO2","H2S","HCN","NH3","C2H2","PH3","H2-He"],
#                       [1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12],
#                       [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
#                       [1e0,1e0,1e0,1e0,1e0,1e0,1e0,1e0,1e0,1e0])
#                       [1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1])
#fit_info.add_gases_vmr(["H2O","CH4","CO2","CO","H2S","HCN","NH3","H2-He"],
#                       [1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12],
#                       [1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1])
#fit_info.add_gases_vmr(["H2O","CH4","CO2","CO","H2S","SO2","NH3","H2-He"],
#                       [1e-12,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12],
#                       [1e-1,1e-1,1e-1,1e-1,1e-1,1e-1,1e-1])
#fit_info.add_gases_vmr(["H2O","CH4","CO2","CO","H2S","NH3","H2-He"],
#                       [1e-12,1e-12,1e-12,1e-12,1e-12,1e-12],
#                       [1e-1,1e-1,1e-1,1e-1,1e-1,1e-1])
#fit_info.add_gases_vmr(["SO2"],10**-12,10**-3)

#Add fitting parameters - this specifies which parameters you want to fit
#e.g. since we have not included cloudtop_P, it will be fixed at the value specified in the constructor

#fit_info.add_gaussian_fit_param('Rs', 0.04*R_sun)
#fit_info.add_gaussian_fit_param('Mp', merr*M_earth)
#fit_info.add_gaussian_fit_param('Mp', 2.2*M_earth)
#fit_info.add_gaussian_fit_param("T_star", 100)
#fit_info.add_gaussian_fit_param("T_spot", 100)
fit_info.add_gaussian_fit_param("T_star", 37)
fit_info.add_gaussian_fit_param("T_spot", 69)
#fit_info.add_uniform_fit_param('Mp', 5.0*M_earth, 15.0*M_earth)
#fit_info.add_uniform_fit_param('Mp', 5*M_earth, 100*M_earth)
#fit_info.add_uniform_fit_param('logMp', np.log10(1*M_earth), np.log10(100*M_earth))
#fit_info.add_gaussian_fit_param('Rp', 0.13*R_earth)

#fit_info.add_uniform_fit_param('Mp', 5*M_earth, 22*M_earth)
fit_info.add_uniform_fit_param('Mp', M_earth, 50*M_earth)
fit_info.add_uniform_fit_param('Rp', 2*R_earth, 5*R_earth)
#fit_info.add_uniform_fit_param('T', 200, 1000)
fit_info.add_uniform_fit_param('T', 200, 1000)
#fit_info.add_uniform_fit_param("log_scatt_factor", 0, 5)
#fit_info.add_uniform_fit_param("scatt_slope", 0, 20)
#fit_info.add_uniform_fit_param("T_star", 3000, 8000)
#fit_info.add_uniform_fit_param("T_spot", 3000, 8000)
fit_info.add_uniform_fit_param("spot_cov_frac", 0, 1)
#fit_info.add_uniform_fit_param("logZ", -1, 3)
fit_info.add_uniform_fit_param("logZ", -1, 3)
#fit_info.add_uniform_fit_param("log_cloudtop_P", -0.99, 5) # 10^parameter Pa
#fit_info.add_uniform_fit_param("log_cloudtop_P", -3.999, 7) # 10^parameter Pa
fit_info.add_uniform_fit_param("error_multiple", 0, np.max(tderr))
fit_info.add_uniform_fit_param("offset_transit", -1000e-6, 1000e-6)
fit_info.add_uniform_fit_param("CO_ratio",0.05,1.5)
#fit_info.add_uniform_fit_param("cloud_frac",0.,1.)

#result = retriever.run_multinest(wvbins[6:,:], td[6:], tderr[6:],
result = retriever.run_multinest(wvbins, td, tderr,
                                 None, None, None,
                                 fit_info, #nwalkers=250, nsteps=5000,
                                 #basename="",
                                 sample="rwalk",  nlive=1000,
                                 rad_method="xsec") #"ktables" to use corr-k
with open(pfolder+'aumicb_nlive1000_contam_newdataDec28_errinfl_widerMpPrior_newT_clear.pkl', "wb") as f:
    pickle.dump(result, f)

plotter = Plotter()
plotter.plot_retrieval_transit_spectrum(result, prefix=pfolder+
                                        'aumicb_nlive1000_contam_newdataDec28_errinfl_widerMpPrior_newT_clear')
plotter.plot_retrieval_corner(result, filename=pfolder+
                              'aumicb_nlive1000_contam_newdataDec28_errinfl_widerMpPrior_newT_clear_corner.pdf')


