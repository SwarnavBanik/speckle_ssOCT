"""
###############################################################################
Code for estimating speckle patterns for OCT applications
###############################################################################
Created:    Swarnav Banik  on  Nov 17, 2024
"""

CODENAME = 'plotSpk_octAscan_NlambdaCompound'
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('../')
from src.frmtFig import frmtFig 
from src.frmtFig import setFigureSize
from src.speckle import octSignal_aScan

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
wA                  = 10E-6        # beam waist [m]
z0A                 = 50E-3        # beam waist position wrt collimator [m]
zTarget             = 50E-3        # target position wrt collimator [m]
deltaZ_col2target   = 500E-6       # axial length of sample [m]
wavelength          = 1050E-9      # wavelength [m]
rxColDiameter       = 12.7E-3      # Collimator Diameter [m]
lensOffset          = [0,0]        # Lens offset wrt target [m] - descan/ pitch catch
Ngrid_target        = 30           # No. of target grid points 
Ngrid_rxCol         = 100           # No. of rxCol grid points
Nz                  = 1         # No. of z grid points
Nlambda             = 9
Nruns               = 1000           # No. of averaging points
journal             = True         # Figure type

# %% Initialize ###############################################################
clrPts, mpl, plt = frmtFig(mpl, plt, FS_title = 24, FS_tickLabel = 18, FS_axisLabel = 18, journal = journal)

# %% Generate a single Speckle Pattern and evaluate it's characteristics ######
start_time = time.time()
aScans     = np.zeros((Nruns, Nlambda+1, Nz + 1 - Nz % 2))

for ii in range(Nruns):  
    signal = 0
    for jj in range(Nlambda+1):        
        a = octSignal_aScan(wA, z0A, zTarget, deltaZ_col2target = deltaZ_col2target, Nz = Nz,\
                            wavelength = wavelength, Ngrid_target = Ngrid_target, phaseScale = 1,\
                            lensDiameter = rxColDiameter, Ngrid_rxCol = Ngrid_rxCol, lensOffset = lensOffset)
        signal = signal + a.aScan
        aScans[ii, jj, :] = signal

end_time = time.time()
print(f"Time taken for {Nruns:d} run(s) of {Nz:d} points: {end_time-start_time:.4f} seconds")
# %% Plot Rayleigh Density statistics of the RX col field #####################
Sfrac_A, PtimesavgS_A, PtimesavgS_theory = getAvgTrace(aScans[:,0,:])
Sfrac_B, PtimesavgS_B, _ = getAvgTrace(aScans[:,1,:])
Sfrac_C, PtimesavgS_C, _ = getAvgTrace(aScans[:,Nlambda,:])

fig = setFigureSize(plt, 7, (12,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(Sfrac_A , PtimesavgS_A,'.', color = clrPts[0], label = 'Simulation - Signals A')
axs1.plot(Sfrac_B , PtimesavgS_B,'.', color = clrPts[1], label = 'Simulation - Signals B')
axs1.plot(Sfrac_C , PtimesavgS_C,'.', color = clrPts[3], label = 'Simulation - Signals C')
axs1.plot(Sfrac_A , PtimesavgS_theory,'-', color = clrPts[0], label = 'Theory')
axs1.legend()
#axs1.grid('major')
axs1.set_ylabel('$\\left<S\\right>$ P(S)')
axs1.set_xlabel('S/$\\left<S\\right>$')
axs1.set(ylim=( -0.1, 3.5 ))
axs1.set(xlim=( 0, 3 ))

plt.savefig(CODENAME+'_dist.svg') 
plt.savefig(CODENAME+'_dist.png') 

# %% Plot the SNR trendline ###################################################
snr = np.zeros((Nlambda+1,))
for ii in range(Nlambda+1):
    snr[ii] = getSNR(aScans[:,ii,:].flatten())

theory = np.sqrt(np.floor(range(1,Nlambda+2))) * snr[0]

    
fig = setFigureSize(plt, 8, (12,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(range(1,Nlambda+2) , 10*np.log10(snr),'.', color = clrPts[0], label = 'Simulation - Signals A')
axs1.plot(range(1,Nlambda+2) , 10*np.log10(theory),'--', color = clrPts[0], label = 'Simulation - Signals A')

axs1.legend()
axs1.grid('major')
axs1.set_ylabel('SNR (dB)')
axs1.set_xlabel('No. of wavelength')
#axs1.set(ylim=( 0, 10 ))
axs1.set(xlim=( 0,  Nlambda+2))

plt.savefig(CODENAME+'_trend.svg') 
plt.savefig(CODENAME+'_trend.png') 

# %% Save the variables #######################################################
import pickle

with open(CODENAME+'.pkl', 'wb') as f:
    pickle.dump({'Sfrac_A': Sfrac_A, 'PtimesavgS_A': PtimesavgS_A, 'PtimesavgS_theory': PtimesavgS_theory,\
                  'Sfrac_B': Sfrac_B, 'PtimesavgS_B': PtimesavgS_B,\
                  'Sfrac_Cmp': Sfrac_C, 'PtimesavgS_C': PtimesavgS_C,\
                  'aScans': aScans,\
                      }, f)

