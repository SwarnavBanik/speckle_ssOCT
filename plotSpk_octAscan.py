"""
###############################################################################
Code for estimating speckle patterns for OCT applications
###############################################################################
Created:    Swarnav Banik  on  Nov 10, 2024
"""
CODENAME = 'plotSpk_octAscan'
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
    hist, bin_edges = np.histogram(field, bins = 100, density = True)
    bin_centers   = (bin_edges[:-1] + bin_edges[1:]) / 2
    averageVal    = np.mean(field)
    Sfrac         = bin_centers/averageVal
    PtimesavgS    = averageVal*hist
    PtimesavgS_theory = np.pi/2 * Sfrac * np.exp( -np.pi/4 * Sfrac**2 )
    return Sfrac, PtimesavgS, PtimesavgS_theory

# %% Inputs ###################################################################
wA                  = 10E-6        # beam waist [m]
z0A                 = 50E-3        # beam waist position wrt collimator [m]
zTarget             = 50E-3        # target position wrt collimator [m]
deltaZ_col2target   = 500E-6       # axial length of sample [m]
wavelength          = 1050E-9      # wavelength [m]
rxColDiameter       = 12.7E-3      # Collimator Diameter [m]
lensOffset          = [0,0]        # Lens offset wrt target [m] - descan/ pitch catch
Ngrid_target        = 30           # No. of target grid points 
Ngrid_rxCol         = 100          # No. of rxCol grid points
Nz                  = 1           # No. of z grid points
Nruns               = 1000            # No. of averaging points
journal             = True        # Figure type

# %% Initialize ###############################################################
clrPts, mpl, plt = frmtFig(mpl, plt, FS_title = 24, FS_tickLabel = 18, FS_axisLabel = 18, journal = journal)

# %% Generate a single Speckle Pattern and evaluate it's characteristics ######
start_time = time.time()
signals = []
for ii in range(Nruns):
    a = octSignal_aScan(wA, z0A, zTarget, deltaZ_col2target = deltaZ_col2target, Nz = Nz,\
                        wavelength = wavelength, Ngrid_target = Ngrid_target, phaseScale = 1,\
                        lensDiameter = rxColDiameter, Ngrid_rxCol = Ngrid_rxCol, lensOffset = lensOffset)
    signals.append( a.aScan )
    if ii == int(Nruns/2):   
        a.plotAscan(0, journal = journal)
        a.plotSpkPatterns_z(1, journal = journal)
end_time = time.time()
print(f"Time taken for {Nruns:d} run(s) of {Nz:d} points: {end_time-start_time:.4f} seconds")
# %% Plot Rayleigh Density statistics of the RX col field #####################
Sfrac, PtimesavgS, PtimesavgS_theory = getAvgTrace(signals)

fig = setFigureSize(plt, 4, (12,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(Sfrac , PtimesavgS,'.', color = clrPts[0], label = 'Simulation')
axs1.plot(Sfrac , PtimesavgS_theory,'-', color = clrPts[1], label = 'Theory')
axs1.legend()
axs1.grid('major')
axs1.set_ylabel('$\\left<S\\right>$ P(S)')
axs1.set_xlabel('S/$\\left<S\\right>$')
axs1.set(ylim=( -0.1, 1.3 ))
axs1.set(xlim=( 0, 3 ))

# %% Save the variables #######################################################
import pickle

with open(CODENAME+'.pkl', 'wb') as f:
    pickle.dump({'Sfrac': Sfrac, 'PtimesavgS': PtimesavgS, 'PtimesavgS_theory': PtimesavgS_theory,\
                  'asignals': signals,\
                      }, f)