"""
###############################################################################
Code for estimating speckle patterns for OCT applications
###############################################################################
Created:    Swarnav Banik  on  Jun 28, 2024
"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


sys.path.append('../')
from src.frmtFig import frmtFig 
from src.frmtFig import setFigureSize
from src.gaussBeam import gaussBeam
from src.gridMnp import gridMap_uniformSym
clrPts, mpl, plt = frmtFig(mpl, plt, FS_title = 18, FS_tickLabel = 18, FS_axisLabel = 18)

# %% Common Functions #########################################################
def lorentzian(x, center, fwhm):
    """
    Lorentzian function.

    Parameters:
        x: Independent variable.
        amplitude: Amplitude of the peak.
        center: Center of the peak.
        width: Full width at half maximum (FWHM).

    Returns:
        Lorentzian function value at x.
    """
    hwhm = fwhm/2
    L = (1/np.pi) * ( hwhm / ( (x-center)**2 + hwhm**2 ) )
    return L
def threeStep(x, x0 = 0, l = 1):
    xStart = x0 - l/2
    xMid1  = xStart + l/3
    xMid2  = xStart + 2*l/3
    xEnd   = x0 + l/2
    y = 0.01*np.ones(np.shape(x))
    
    for ii in range(len(x)):    
        if x[ii] < xStart:
            y[ii] = 0
        elif x[ii] >= xStart and x[ii] < xMid1:
            y[ii] = 1/3
        elif x[ii] >= xMid1 and x[ii] < xMid2:
            y[ii] = 2/3
        elif x[ii] >= xMid2 and x[ii] <= xEnd:
            y[ii] = 1
    return y
        
    

# %% Master Speckle Pattern ###################################################
class specklePattern:
    """
    Speckle Pattern imparted to any field
    Inputs:
        field_b4spk:   Field before speckle randomization is applied
        grid_target:   2D Grid Object of the target surface 
        phaseScale:    Extent of phase randomization [0 to 1]
        rndPhase:      Optional. If you want a specific phase to be imparted.
    """
    def __init__(self, field_b4spk, grid_target:gridMap_uniformSym, wavelength = 1310E-9, phaseScale = 0, rndPhase = []):  
        if len(np.shape(field_b4spk)) != 2:
            raise Exception('specklePattern: The speckle field needs to be 2 dimensional.')
        if not np.shape(field_b4spk) == np.shape(grid_target.X):
            raise Exception('specklePattern: The speckle field and grid need to be of the same size.')
        self.wavelength      = wavelength
        self.field_beforeSpk = field_b4spk
        self.grid_target     = grid_target
            
        # Random phase imparted at the lambertzian surface ####################
        if rndPhase == []:
            self.phase = 2*np.pi*phaseScale*np.random.rand(self.field_beforeSpk.shape[0], self.field_beforeSpk.shape[1])  
            self.phase = self.phase - np.pi
            # self.phase = gaussian_filter(self.phase, sigma=100)            
            # self.phase = self.grid_target.modR*np.pi/np.max(self.grid_target.modR)
            # self.phase = self.grid_target.modR*0
        elif np.shape(rndPhase) == np.shape(self.grid_target.X):            
            self.phase = rndPhase
        else:
            raise Exception('specklePattern: The random phase and grid need to be of the same size.')
        self.field_afterSpk = self.field_beforeSpk * np.exp(-1j * self.phase)

    def fieldPropagation(self, r_rx = [0,0]):
        if len(r_rx) != 2:
            raise Exception('specklePattern::fieldPropagation: Length of l_max needs to be same as the dimesnion of the grid.')
        
        r12 = np.sqrt( (self.grid_target.X-r_rx[0])**2 \
                     + (self.grid_target.Y-r_rx[1])**2 \
                     +  self.zPos_target2rxCol**2 ) 
        propMatrix = self.zPos_target2rxCol / (1j*self.wavelength) * np.exp(-1j*(2*np.pi/self.wavelength)*r12) / r12**2 \
            * np.abs(self.grid_target.X[0,1]-self.grid_target.X[0,0])*np.abs(self.grid_target.Y[1,0]-self.grid_target.Y[0,0])
        rx = np.sum(self.field_afterSpk * propMatrix)
        phaseBaseline = np.sum( self.field_beforeSpk * propMatrix )
        return rx, phaseBaseline
    
    def propagate(self, z, lensDiameter = 18E-3, Ngrid = 10, lensOffset = [0,0]):
        if len(lensOffset) != 2:
            raise Exception('specklePattern::propagate: Length of lensOffset needs to be same as the dimesnion of the grid.')
        self.lensDiameter = lensDiameter
        self.lensOffset   = lensOffset
        
        self.zPos_target2rxCol  = z
        self.grid_rxCol = gridMap_uniformSym(2, [self.lensDiameter/2, self.lensDiameter/2], [Ngrid, Ngrid])

        self.field_rxCol            = np.zeros(np.shape(self.grid_rxCol.X), dtype = 'complex')
        self.field_rxCol_baseline   = np.zeros(np.shape(self.grid_rxCol.X), dtype = 'complex')
        for ii in range(Ngrid):
            for jj in range(Ngrid):
                self.field_rxCol[ii,jj], self.field_rxCol_baseline[ii,jj]  = self.fieldPropagation(\
                                    r_rx = [self.grid_rxCol.X[ii,jj]+self.lensOffset[0], \
                                    self.grid_rxCol.Y[ii,jj]+self.lensOffset[1]])
        
        # Evaluate statistics #################################################
        fld            = np.abs(self.field_rxCol.flatten()) 
        averageVal     = np.mean(fld)
        hist, bin_edges = np.histogram(fld, bins = 100, density = True)
        bin_centers   = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.rxCol_Sfrac      = bin_centers/averageVal
        self.rxCol_PtimesavgS = averageVal*hist
        self.rxCol_PtimesavgS_theory = np.pi/2 * self.rxCol_Sfrac \
            * np.exp( -np.pi/4 * self.rxCol_Sfrac**2 )
        
        
                                        
    def plotField(self, fig, axs1, axs2, field, grid:gridMap_uniformSym, units = 'mm'):
        if units =='mm':
            c = axs1.pcolor(grid.X[0,:]*1e3,grid.Y[:,0]*1e3, \
                          (np.abs(field)/np.max(np.abs(field)))**2,\
                          cmap = 'Reds', vmin=0, vmax=1)
            axs1.set_xlabel('X (mm)')
            axs1.set_ylabel('Y (mm)')
        elif units == 'um':
            c = axs1.pcolor(grid.X[0,:]*1e6,grid.Y[:,0]*1e6, \
                          (np.abs(field)/np.max(np.abs(field)))**2,\
                          cmap = 'Reds', vmin=0, vmax=1)
            axs1.set_xlabel('X ($\\mu$m)')
            axs1.set_ylabel('Y ($\\mu$m)')
        else:
            raise Exception('specklePattern::plotField: invalid units.')
        cb1 = fig.colorbar(c, ax=axs1, label = 'Beam Intensity', location='bottom')
        cb1.set_label(label='Normalized Beam Intensity')
        
        c = axs2.pcolor(grid.X[0,:]*1e3,grid.Y[:,0]*1e3, \
                      np.angle(field)/np.pi, cmap = 'Blues',\
                      vmin=-1, vmax=1)
        cb2 = fig.colorbar(c, ax=axs2, label = 'Phase', location='bottom', ticks=[-1, 0, 1])
        cb2.ax.set_xticklabels(['$-\\pi$','0', '$\\pi$']) 
        cb2.set_label(label='Phase') 
        axs2.set_xlabel('X (mm)')
        return fig, axs1, axs2, cb1, cb2

    def plotSingleField(self, figNo, field, grid:gridMap_uniformSym, journal = False):
        fig = setFigureSize(plt, figNo, (18,12), journal = journal)
        gs = GridSpec(1,2)
        fig.clf()
        axs1 = fig.add_subplot(gs[0,0]) 
        axs2 = fig.add_subplot(gs[0,1]) 
        if np.abs(grid.X[-1,-1]-grid.X[0,0]) < 500E-6:
            units = 'um'
        else:
            units = 'mm'
        fig, axs1, axs2, cb1, cb2 = self.plotField(fig, axs1, axs2, field, grid, units=units)
        return fig, axs1, axs2, cb1, cb2
    
    def changePhase(self, fig, axs, field, grid:gridMap_uniformSym):
        c = axs.pcolor(grid.X[0,:]*1e3,grid.Y[:,0]*1e3, \
                      np.angle(field)/np.pi, cmap = 'Blues',\
                      vmin=-1, vmax=1)
        cb = fig.colorbar(c, ax=axs, label = 'Phase', location='bottom', ticks=[-1, 0, 1])
        cb.ax.set_xticklabels(['$-\\pi$','0', '$\\pi$']) 
        cb.set_label(label='Phase') 
        return fig, axs     
    
    def printSingleField(self, figNo, field = 'at target after speckle', limits = [], journal = False):
        if field == 'at target before speckle':
            field = self.field_beforeSpk
            grid  = self.grid_target
            title = '\n Field at Target before Speckle'
        elif field == 'at target after speckle':
            field = self.field_afterSpk
            grid  = self.grid_target
            title = '\n Field at Target after Speckle'
        elif field == 'at collimator':
            field = self.field_rxCol 
            grid  = self.grid_rxCol
            title = '\n Field at RX colimator'
        else:
            raise Exception('specklePattern_gauss::printField: Invalid field.')
        fig, axs1, axs2, cb1, cb2 = self.plotSingleField(figNo, field, grid, journal = journal)
        
        if title == '\n Field at RX colimator':
            cb2.remove()
            fig, axs2 = self.changePhase(fig, axs2, self.field_rxCol * np.conj(self.field_rxCol_baseline),\
                                          self.grid_rxCol)          
            circle = plt.Circle(self.lensOffset, self.lensDiameter*1E3/2 , ls = '--',\
                                color=(0,0,0), fill = False)
            axs1.add_patch(circle)
            circle = plt.Circle(self.lensOffset, self.lensDiameter*1E3/2 , ls = '--',\
                                color=(0,0,0), fill = False)
            axs2.add_patch(circle)
            
       
        fig.suptitle(title)
        if not limits == []:
            axs1.set(xlim = limits)
            axs1.set(ylim = limits)
            axs2.set(xlim = limits)
            axs2.set(ylim = limits)
        return fig, axs1, axs2
        
# %% Speckle Pattern for a uniform Gaussian beam ##############################
class specklePattern_gauss:
    """
    Speckle Pattern imparted to a uniform gaussian beam
    Inputs:
        beamWaist:     Beam waist radius [m]
        z_col2waist:   Beam waist position wrt to collimator [m]
        z_col2target:  Distance between collimator and target [m]
        phaseScale:    Extent of phase randomization [0 to 1]
        rndPhase:      Optional. If you want a specific phase to be imparted.
    """
    def __init__(self, beamWaist, z_col2waist, z_col2target, wavelength = 1310E-9, \
                 Ngrid = 1000, phaseScale = 1, rndPhase = []): 
        
        # Define the coordinate system with origin at the colimator ###########
        self.beam_tx     = gaussBeam( beamWaist, z_col2waist, wavelength = wavelength, M2 = 1)
        self.zPos_target = z_col2target
        beamWidth_target = self.beam_tx.gaussBeamWaist(self.zPos_target)
        grid_target      = gridMap_uniformSym(2, [3.5*beamWidth_target, 3.5*beamWidth_target], [Ngrid, Ngrid])
        field_target     = self.beam_tx.gaussBeamField(grid_target.X, grid_target.Y, self.zPos_target,0,0)
        
        self.spkPattern  = specklePattern(field_target, grid_target, wavelength = self.beam_tx.wavelength, \
                                          phaseScale = phaseScale, rndPhase = rndPhase)
        
        
    def propagate(self, z_target2col, lensDiameter = 18E-3, Ngrid = 50, lensOffset = [0,0]):
        self.spkPattern.propagate(z_target2col, lensDiameter = lensDiameter, Ngrid = Ngrid, lensOffset = lensOffset)
        
    def plotFieldOverlap(self, figNo, field1, field2, grid:gridMap_uniformSym, journal = False):
        fig = setFigureSize(plt, figNo, (18,12*2), journal = journal)
        gs = GridSpec(2,2)
        fig.clf()
        axs1 = fig.add_subplot(gs[0,0]) 
        axs2 = fig.add_subplot(gs[0,1]) 
        axs3 = fig.add_subplot(gs[1,0]) 
        axs4 = fig.add_subplot(gs[1,1]) 
        fig, axs1, axs2, _,_ = self.spkPattern.plotField(fig, axs1, axs2, field1, grid)
        circle = plt.Circle(self.spkPattern.lensOffset, self.spkPattern.lensDiameter*1E3/2 , ls = '--',\
                            color=(0,0,0), fill = False)
        axs1.add_patch(circle)
        circle = plt.Circle(self.spkPattern.lensOffset, self.spkPattern.lensDiameter*1E3/2 , ls = '--',\
                            color=(0,0,0), fill = False)
        axs2.add_patch(circle)
        fig, axs3, axs4, _,_  = self.spkPattern.plotField(fig, axs3, axs4, field2, grid)
        circle = plt.Circle(self.spkPattern.lensOffset, self.spkPattern.lensDiameter*1E3/2 , ls = '--',\
                            color=(0,0,0), fill = False)
        axs3.add_patch(circle)
        circle = plt.Circle(self.spkPattern.lensOffset, self.spkPattern.lensDiameter*1E3/2 , ls = '--',\
                            color=(0,0,0), fill = False)
        axs4.add_patch(circle)
        title = '\n RX Field Overlap'
        fig.suptitle(title)
        return fig, axs1, axs2, axs3, axs4
          
    def modeOvrlp_rxCol(self, printFig = True, figNo = 3, journal = False):
        if not hasattr(self.spkPattern, 'field_rxCol'):
            raise Exception('specklePattern_gauss::modeOvrlp_rxCol: Propagate the speckle field to the RX colimator first')
        else:
            dx = np.abs( self.spkPattern.grid_rxCol.X[0,1] - self.spkPattern.grid_rxCol.X[0,0] )
            dy = np.abs( self.spkPattern.grid_rxCol.Y[1,0] - self.spkPattern.grid_rxCol.Y[0,0] )
            field1 = self.spkPattern.field_rxCol
            field2 = self.beam_tx.gaussBeamField(self.spkPattern.grid_rxCol.X, self.spkPattern.grid_rxCol.Y, 0, 0, 0)
            idx = np.sqrt(self.spkPattern.grid_rxCol.X**2 + self.spkPattern.grid_rxCol.Y**2) > self.spkPattern.lensDiameter/2
            field2[idx] = 0
                                  
            overlap_integral = np.sum( np.conjugate(field1) * field2) * dx * dy
            norm1            = np.sqrt(np.sum(np.abs(field1)**2) * dx * dy)
            norm2            = np.sqrt(np.sum(np.abs(field2)**2) * dx * dy)
            
            if printFig:
                self.plotFieldOverlap(figNo,field1, field2, self.spkPattern.grid_rxCol, journal = journal)
                     
            return np.abs( overlap_integral / (norm1 * norm2) )
        
    def plotFieldOverlap_rxCol(self,  figNo = 3, journal = False):
        if not hasattr(self.spkPattern, 'field_rxCol'):
            raise Exception('specklePattern_gauss::modeOvrlp_rxCol: Propagate the speckle field to the RX colimator first')
        else:
            field1 = self.spkPattern.field_rxCol
            field2 = self.beam_tx.gaussBeamField(self.spkPattern.grid_rxCol.X, self.spkPattern.grid_rxCol.Y, 0, 0, 0)
            fig, axs1, axs2, axs3, axs4 = self.plotFieldOverlap(figNo,field1, field2, self.spkPattern.grid_rxCol, journal = journal)
            
            return fig, axs1, axs2, axs3, axs4

# %% OCT A scan signal ########################################################

class octSignal_aScan:
    """
    OCT A scan Signal with a Gaussian Beam
    Inputs:
        beamWaist:     Beam waist radius [m]
        z_col2waist:   Beam waist position wrt to collimator [m]
        z_col2target:  Distance between collimator and target [m]
        phaseScale:    Extent of phase randomization [0 to 1]
        rndPhase:      Optional. If you want a specific phase to be imparted.
    """
    def __init__(self, beamWaist, z_col2waist, z_col2target, \
                 deltaZ_col2target = 500E-6, Nz = 11,\
                 wavelength = 1310E-9, Ngrid_target = 1000, phaseScale = 1, rndPhase = [],\
                 lensDiameter = 18E-3, Ngrid_rxCol = 50, lensOffset = [0,0]): 
        
        if Nz % 2 == 0:
            Nz = Nz+1

        # Define the coordinate system with origin at the colimator ###########
        self.beam_tx       = gaussBeam( beamWaist, z_col2waist, wavelength = wavelength, M2 = 1)
        self.zPos_target   = z_col2target
        self.deltaZ_target = deltaZ_col2target
        self.z             = np.linspace(self.zPos_target-self.deltaZ_target/2,\
                                         self.zPos_target+self.deltaZ_target/2,Nz)
        # Go through all Z and determine the signal and equivalent w/o speckle  ##########################
        self.spkPatterns  = []
        self.G            = np.zeros((Nz,))
        self.G_noSpk      = np.zeros((Nz,))       
        for ii in range(Nz):
            a = specklePattern_gauss(beamWaist, z_col2waist, self.z[ii], \
                                      wavelength = wavelength, Ngrid = Ngrid_target, phaseScale = phaseScale)
            a.propagate(self.z[ii], lensDiameter = lensDiameter, Ngrid = Ngrid_rxCol, lensOffset = lensOffset)  
            self.G[ii] = a.modeOvrlp_rxCol(printFig = False)      
            self.spkPatterns.append(a)
            b = specklePattern_gauss(beamWaist, z_col2waist, self.z[ii], \
                                      wavelength = wavelength, Ngrid = Ngrid_target, phaseScale = 0)
            b.propagate(self.z[ii], lensDiameter = lensDiameter, Ngrid = Ngrid_rxCol, lensOffset = lensOffset)
            self.G_noSpk[ii] = b.modeOvrlp_rxCol(printFig = False)
        
        self.aScan     = self.G       
        averageVal     = np.mean(self.aScan)
        hist, bin_edges = np.histogram(self.aScan, bins = 100, density = True)
        bin_centers   = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.Sfrac      = bin_centers/averageVal
        self.PtimesavgS = averageVal*hist
        self.PtimesavgS_theory = np.pi/2 * self.Sfrac * np.exp( -np.pi/4 * self.Sfrac**2 )
            
            
    def plotAscan(self, figNo, journal = False):
        fig = setFigureSize(plt, figNo, (12, 6), journal = journal)
        gs = GridSpec(1,1)
        fig.clf()
        axs1 = fig.add_subplot(gs[0,0]) 
        axs1.plot((self.z-self.zPos_target)*1E6, self.aScan/self.G_noSpk, '.')
        axs1.grid('major')
        axs1.set_xlabel('z - z$_{\\rm obj}$ ($\\mu$m)')
        axs1.set_ylabel('I$_{\\rm a}(\phi) / I_{\\rm a}(\phi=0)$')
        title = 'OCT A scan Signal'
        fig.suptitle(title)
        return fig, axs1
            
    def plotAscanStats(self, figNo, journal = False):
        fig = setFigureSize(plt, figNo, (12, 6), journal = journal)
        gs = GridSpec(1,1)
        fig.clf()
        axs1 = fig.add_subplot(gs[0,0]) 
        axs1.plot(self.Sfrac, self.PtimesavgS, '.')
        axs1.plot(self.Sfrac, self.PtimesavgS_theory, '-')
        axs1.grid('major')
        axs1.set_ylabel('$\\left<S\\right>$ P(S)')
        axs1.set_xlabel('S/$\\left<S\\right>$')
        title = 'A-scan Statistics'
        fig.suptitle(title)
        return fig, axs1
    
    def plotSpkPatterns_modeOverlap(self, figNo, journal = False):
        spkPattern2plot = self.spkPatterns[0]
        fig,_,_,_,_ = spkPattern2plot.plotFieldOverlap_rxCol(figNo = figNo, journal = journal)
        fig.suptitle(f'\nRX Field Overlap \n z = {self.z[0]*1E3:0.3f} mm')
        spkPattern2plot = self.spkPatterns[len(self.spkPatterns)//2]
        fig,_,_,_,_ = spkPattern2plot.plotFieldOverlap_rxCol(figNo = figNo+1, journal = journal)
        fig.suptitle(f'\nRX Field Overlap \n z = {self.z[len(self.spkPatterns)//2]*1E3:0.3f} mm')
        spkPattern2plot = self.spkPatterns[-1]
        fig,_,_,_,_ = spkPattern2plot.plotFieldOverlap_rxCol(figNo = figNo+2, journal = journal)
        fig.suptitle(f'\nRX Field Overlap \n z = {self.z[-1]*1E3:0.3f} mm')

    def plotSpkPatterns_z(self, figNo, z = [], journal = False):
        if z == []:
            z = self.zPos_target
        idx = np.argmin(np.abs(self.z - z))
        spkPattern2plot = self.spkPatterns[idx]
        spkPattern2plot.spkPattern.printSingleField(figNo+0, field = 'at target before speckle', journal = journal)
        spkPattern2plot.spkPattern.printSingleField(figNo+1, field = 'at target after speckle', journal = journal)
        spkPattern2plot.spkPattern.printSingleField(figNo+2, field = 'at collimator', limits = [], journal = journal)
        
