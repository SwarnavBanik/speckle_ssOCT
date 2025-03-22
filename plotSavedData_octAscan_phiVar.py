"""
###############################################################################
Make Fig 3 from saved data
###############################################################################
Created:    Swarnav Banik  on  Nov 17, 2024
"""

CODENAME = 'plotSavedData_octAscan_phiVar'
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('../')
from src.frmtFig import frmtFig 
from src.frmtFig import setFigureSize


import pickle
pickle_file = 'savedData/plotSpk_octAscan_phiVar.pkl'

# %% Common Functions #########################################################
def getAvgTrace(field):
    hist, bin_edges = np.histogram(field, bins = 70, density = True)
    bin_centers   = (bin_edges[:-1] + bin_edges[1:]) / 2
    averageVal    = np.mean(field)
    Sfrac         = bin_centers/averageVal
    PtimesavgS    = averageVal*hist
    PtimesavgS_theory = np.pi/2 * Sfrac * np.exp( -np.pi/4 * Sfrac**2 )
    return Sfrac, PtimesavgS, PtimesavgS_theory

def getContrast(signal):
    return np.std(signal)/np.mean(signal)

def getSNR(signal):
    return np.mean(signal)/np.std(signal)

# %% Inputs ###################################################################
journal             = True         # Figure type

# %% Initialize ###############################################################
clrPts, mpl, plt = frmtFig(mpl, plt, FS_title = 24, FS_tickLabel = 18, FS_axisLabel = 18, journal = journal)


# %% Load the values and make plots ###########################################

with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
    
fig = setFigureSize(plt, 5, (12,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(data['Sfrac_b'] , data['PtimesavgS_b'],'.', color = clrPts[1], label = '$\delta \phi \in 0.3~[-\pi, \pi]$')
axs1.plot(data['Sfrac_c'] , data['PtimesavgS_c'],'.', color = clrPts[2], label = '$\delta \phi \in 0.7~[-\pi, \pi]$')
axs1.plot(data['Sfrac_d'] , data['PtimesavgS_d'],'.', color = clrPts[3], label = '$\delta \phi \in 1.0~[-\pi, \pi]$')
axs1.plot(data['Sfrac_d'] , data['PtimesavgS_theory'],'-', color = clrPts[3], label = 'Theory')
axs1.legend()
axs1.grid('major')
axs1.set_ylabel('$\\left<I_{\\rm a} \\right>$ P($I_{\\rm a}$)')
axs1.set_xlabel('$I_{\\rm a}/ \\left<I_{\\rm a}\\right>$')
axs1.legend()
axs1.set(ylim=( 0, 6 ))
# axs1.set(xlim=( 0, 3 ))

SNR_b = 10*np.log10(getSNR(data['signals_b']))
SNR_c = 10*np.log10(getSNR(data['signals_c']))
SNR_d = 10*np.log10(getSNR(data['signals_d']))


