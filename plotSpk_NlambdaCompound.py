"""
###############################################################################
Code for estimating N uncorrelated speckle patterns for OCT applications
###############################################################################
Created:    Swarnav Banik  on  Nov 19, 2024
"""
CODENAME = 'plotSpk_NlambdaCompound'
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('../')
from src.frmtFig import frmtFig 
from src.frmtFig import setFigureSize
from src.speckle import specklePattern_gauss

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
wA             = 10E-6           # beam waist [m]
z0A            = 50E-3           # beam waist position wrt collimator [m]
zTarget        = 50E-3           # target position wrt collimator [m]
wavelength     = 1050E-9         # wavelength 1 [m]
rxColDiameter  = 12.7E-3         # Collimator Diameter [m]
lensOffset     = [0,0]           # Lens offset wrt target [m] - descan/ pitch catch
Ngrid_target   = 30              # No. of target grid points 
Ngrid_rxCol    = 100              # No. of rxCol grid points
Nlambda        = 9              # No. of uncorrelated signals
Nruns          = 10              # No. of averaging points
journal        = True            # figure type

# %% Initialize ###############################################################
clrPts, mpl, plt = frmtFig(mpl, plt, FS_title = 24, FS_tickLabel = 18, FS_axisLabel = 18, journal = journal)

# %% Generate a single Speckle Pattern and evaluate it's characteristics ######
start_time = time.time()
overlapIntg = np.zeros((Nruns,1))
fields = []

fields     = np.zeros((Nruns, Nlambda+1, Ngrid_rxCol*Ngrid_rxCol))

for ii in range(Nruns):
    signal = np.zeros((Ngrid_rxCol,Ngrid_rxCol))
    for jj in range(Nlambda+1): 
        a = specklePattern_gauss(wA, z0A, zTarget, wavelength = wavelength, \
                                 Ngrid = Ngrid_target, phaseScale = 1)
        a.propagate(zTarget, lensDiameter = rxColDiameter, Ngrid = Ngrid_rxCol, \
                    lensOffset = lensOffset)
        signal = signal + np.abs(a.spkPattern.field_rxCol)
        
        fields[ii, jj, :] = signal.flatten()
end_time = time.time()
print(f"Time taken for {Nruns:d} run(s): {end_time-start_time:.4f} seconds")

# %% Plot Rayleigh Density statistics of the RX col field #####################
Sfrac_A, PtimesavgS_A, PtimesavgS_theory = getAvgTrace(fields[:,0,:])
Sfrac_B, PtimesavgS_B, _ = getAvgTrace(fields[:,1,:])
Sfrac_C, PtimesavgS_C, _ = getAvgTrace(fields[:,Nlambda,:])

fig = setFigureSize(plt, 7, (6,6), journal = journal)
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
axs1.set(ylim=( -0.1, 3 ))
axs1.set(xlim=( 0, 2 ))

plt.savefig(CODENAME+'_dist.svg') 
plt.savefig(CODENAME+'_dist.png') 

# %% Plot the SNR trendline ###################################################
snr = np.zeros((Nlambda+1,))
for ii in range(Nlambda+1):
    snr[ii] = getSNR(fields[:,ii,:].flatten())
theory_x = np.linspace(1,10,100)
theory = np.sqrt(theory_x) * 10**(2.84/10)

fig = setFigureSize(plt, 8, (12,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(range(1,Nlambda+2) , 10*np.log10(snr),'.', color = clrPts[0], label = 'Simulation - Signals A')
axs1.plot(theory_x , 10*np.log10(theory),'-', color = clrPts[0], label = 'Simulation - Signals A')

axs1.legend()
axs1.grid('major')
axs1.set_ylabel('SNR (dB)')
axs1.set_xlabel('No. of wavelength')
axs1.set(ylim=( 0, 10 ))
axs1.set(xlim=( 0,  Nlambda+2))

plt.savefig(CODENAME+'_trend.svg') 
plt.savefig(CODENAME+'_trend.png') 
