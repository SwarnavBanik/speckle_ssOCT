### Repo for simulations in "Speckle Noise Mitigation via Incoherent Averaging in Swept Source Optical Coherence Tomography"

The code *plotSpk_octAscan.py* generates the cross-sectional scattered field contribution $\textbf{F}(\textbf{r}, z')$ like the ones shown in Fig. 2 of the manuscript. It propagates a Gaussian beam from a collimator to a target, randomizes its phase upon reflection at the target, and then propagates it back to the collimator.

The code *plotSpk_octAscan_phiVar.py* generates $I_{\rm a}(z)$ for $\delta \phi \in 0.5~[-\pi, \pi)$, $\delta \phi \in 0.7~[-\pi, \pi)$, and $\delta \phi \in [-\pi, \pi)$. It runs each of the three simulations 1000 times and displays their statistical distributions as in Fig. 3 of the manuscript. Since the code *plotSpk_octAscan_phiVar.py* takes a while to run, we provide a saved dataset to quickly plot Fig. 3. The code *plotSaveData_octAscan_phiVar.py* uses this saved dataset to plot Fig. 3.

The code *plotSpk_NlambdaCompound.py* generates $\textbf{F}(\textbf{r}, z')$ for a fully developed speckle for $N_{\rm sig}$ = 1 to 10 and displays their statistical distributions as in Fig. 4(a) of the manuscript. The code *plotSpk_octAscan_NlambdaCompound.py* generates $I_{\rm a}(z)$ for a fully developed speckle for $N_{\rm sig}$ = 1 to 10 . It runs each of the N simulations 1000 times and displays their statistical distributions as in Fig. 4(b) of the manuscript. It also generates Fig. 4(c) of the manuscript. Since the code *plotSpk_octAscan_NlambdaCompound.py* takes about 5 hours to run, we provide a saved dataset to quickly plot Fig. 4(b) and (c). The code *plotSaveData_octAscan_NlambdaCompound.py* uses this saved dataset to plot Fig. 4(b) and (c).
