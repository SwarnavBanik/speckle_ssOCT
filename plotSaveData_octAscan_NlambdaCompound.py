"""
###############################################################################
Make Fig 4(b) and (c) from saved data
###############################################################################
Created:    Swarnav Banik  on  Nov 17, 2024
"""

CODENAME = 'plotSavedData_octAscan_NlambdaCompound'
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('../')
from src.frmtFig import frmtFig 
from src.frmtFig import setFigureSize

import pickle
pickle_file = 'savedData/plotSpk_octAscan_NlambdaCompound.pkl'

# %% Common Functions #########################################################
def getAvgTrace(field):
    field = field.flatten()
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
Nlambda             = 9
journal             = True         

# %% Load the variables from the pickle file
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
 
clrPts, mpl, plt = frmtFig(mpl, plt, FS_title = 24, FS_tickLabel = 18, FS_axisLabel = 18, journal = journal)

    
Sfrac_A, PtimesavgS_A, PtimesavgS_theory = getAvgTrace(data['aScans'][:,0,:])
Sfrac_B, PtimesavgS_B, _ = getAvgTrace(data['aScans'][:,1,:])
Sfrac_C, PtimesavgS_C, _ = getAvgTrace(data['aScans'][:,Nlambda,:])

fig = setFigureSize(plt, 7, (6,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(Sfrac_A , PtimesavgS_A,'.', color = clrPts[0], label = 'Simulation - Signals A')
axs1.plot(Sfrac_B , PtimesavgS_B,'.', color = clrPts[1], label = 'Simulation - Signals B')
axs1.plot(Sfrac_C , PtimesavgS_C,'.', color = clrPts[3], label = 'Simulation - Signals C')
axs1.plot(Sfrac_A , PtimesavgS_theory,'-', color = clrPts[0], label = 'Theory')
axs1.legend()
#axs1.grid('major')
axs1.set_ylabel('$\\left<I_{\\rm a} \\right>$ P($I_{\\rm a}$)')
axs1.set_xlabel('$I_{\\rm a}/ \\left<I_{\\rm a}\\right>$')
axs1.set(ylim=( -0.1, 3.0 ))
axs1.set(xlim=( 0, 2 ))



snr = np.zeros((Nlambda+1,))
for ii in range(Nlambda+1):
    snr[ii] = getSNR(data['aScans'][:,ii,:].flatten())
theory_x = np.linspace(1,10,100)
theory = np.sqrt(theory_x) * 10**(2.84/10)
    
fig = setFigureSize(plt, 8, (12,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(range(1,Nlambda+2) , 10*np.log10(snr),'.', color = clrPts[0], label = 'Simulation - Signals A')
axs1.plot(theory_x , 10*np.log10(theory),'--', color = clrPts[0], label = 'Simulation - Signals A')

axs1.legend()
axs1.grid('major')
axs1.set_ylabel('SNR (dB)')
axs1.set_xlabel('N$_{\\rm sig}$')
axs1.set(ylim=( 0, 10 ))
axs1.set(xlim=( 0,  Nlambda+2))

