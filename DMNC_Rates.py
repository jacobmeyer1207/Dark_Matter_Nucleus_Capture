################################################################################
# ENVIRONMENT
################################################################################

import numpy as np
# Bessel functions appear as the wavefunctions here
from scipy.special import spherical_jn,spherical_yn,jv,jvp,yvp
# Numerical integration
from scipy.integrate import quad
# Non-linear equation solver
from scipy.optimize import fsolve
# scipy can't do real order bessel function zeroes, so use mpmath
from mpmath import besseljzero
# binomial coefficients that appear in spherical harmonics products
from scipy.special import comb
# for parallelization
import multiprocessing as mp


# MARK: Class **************************************************************************
class Rates:
    def __init__(self, Z = 18, A = 40, mu_mult = .938, V0_mult = .246, velDM = .001, R = 10.0):
    ################################################################################
    # PARAMETERS
    ################################################################################
        
        self.Z = Z                                  # material atomic number
        self.A = A                                  # material mass number
        self.e = np.sqrt(4.0 * np.pi / 137)         # electric charge, dimensionless
        self.mu = A * mu_mult                       # nucleus (reduced) mass, GeV
        self.V0 = A * V0_mult                       # potential depth, GeV (A* expectation value of the higgs)
        self.k = velDM * self.mu                    # incoming DM momentum, GeV (What? mu is the Argon's reduced mass... TODO: check this vs all new uses)
        self.radius = R                             # DM radius, GeV^-1
        self.levels = {}                            # cache for energy levels
        # For States**************************************************************
        self.kapS = np.sqrt(self.k**2 + 2.0 * self.mu * self.V0)   # interior momentum for scattering, GeV
        # For Integrals***********************************************************
        self.rad_int_B_cache = {}


    # Derivatives of spherical bessel functions
    def spherical_jnp(self,l,x):
        return -0.5 * spherical_jn(l,x) / x + np.sqrt(0.5 * np.pi / x) * jvp(l+0.5,x)
    def spherical_ynp(self,l,x):
        return -0.5 * spherical_yn(l,x) / x + np.sqrt(0.5 * np.pi / x) * yvp(l+0.5,x)
    # Spherical bessel function zeros as floats (no mpmath)
    def spherical_jnz(self,l,n):
        return float(besseljzero(l+0.5,n))


    ################################################################################
    # STATES
    ################################################################################

    # momentum inside potential well for bound state, GeV
    def kapB(self,n,l):
        return self.spherical_jnz(l,n) / self.radius
    # (positive) binding energy for state, GeV
    def EB(self,n,l):
        try:
            return self.levels[(n,l)]
        except KeyError:
            res = self.V0 - 0.5 * self.kapB(n,l)**2 / self.mu
            if res > 0.:
                self.levels[(n,l)] = res
            return res
    # emitted photon energy/momentum, GeV
    def q(self,ni,li,nf,lf):
        try:
            return self.levels[(ni,li,nf,lf)]
        except KeyError:
            E_ph = - self.EB(ni,li) + self.EB(nf,lf)
            if E_ph <= 0.:
               raise ValueError('Attempting transition from lower energy state to higher energy state')
            self.levels[(ni,li,nf,lf)] = E_ph
        return E_ph
        '''FIXME! No reason to call EB so many times. Call it once, store the value, test the value, and then return it.'''

    # Maximum n allowed to have bound states below top of potential
    def nmax(self,l,Emin):
        res = int(np.ceil((np.sqrt(2.0*self.mu*self.V0)*self.radius)/np.pi - 0.5 * l))
        while res > 0 and self.EB(res,l) < Emin:
            res -= 1
        while res + 1 > 0 and self.EB(res+1,l) > Emin:
            res += 1
        return res

    # normalization for bound state, GeV^-3/2
    def NB(self,n,l):
        normint = -0.25*(np.pi*self.radius**3*jv(-0.5 + l,self.spherical_jnz(l,n))*jv(1.5 + l,self.spherical_jnz(l,n)))/self.spherical_jnz(l,n)
        return 1.0/np.sqrt(normint)
    # boundary conditions for scattering state
    def bcs(self,Ns,delta,l):
        return ((np.cos(delta) * spherical_jn(l,self.k*self.radius) + np.sin(delta) * spherical_yn(l,self.k*self.radius)) - Ns * spherical_jn(l,self.kapS*self.radius),
                (np.cos(delta) * self.k * self.spherical_jnp(l,self.k*self.radius) + np.sin(delta) * self.k * self.spherical_ynp(l,self.k*self.radius)) - Ns * self.kapS * self.spherical_jnp(l,self.kapS*self.radius))
    # solve boundary conditions to get interior normalization, dimensionless
    def NS(self,l):
        return 4.0*np.pi*1j**l * fsolve(lambda x : self.bcs(x[0],x[1],l),(1.0,0.3))[0]

    # interior wave function profiles for scattering and bound states, dimensionless
    def RS(self,r,l):
        return spherical_jn(l,self.kapS*r)
    def RB(self,r,n,l):
        return spherical_jn(l,self.kapB(n,l)*r)

    ################################################################################
    # INTEGRALS
    ################################################################################

    # general radial integral
    def rad_int(self,Ri,kapi,li,Rf,kapf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        if kapi*self.radius < approx_threshold or kapf*self.radius < approx_threshold or force_full:
            print('Calculating full radial integral... may be slow')
            subinterval_lim = max(50,int(np.ceil(max(kapi*self.radius,kapf*self.radius)/subinterval_periods)))
            rad_int_full = quad(lambda r : Ri(r)*Rf(r)*r**3,0,self.radius,limit=subinterval_lim)
            if rad_int_full[1] > 0.01 * abs(rad_int_full[0]):
                print('Error on radial integral greater than 1%')
            res = rad_int_full[0]
        else:
            kap_sum = kapi + kapf
            kap_dif = kapi - kapf
            kap_sum_R = kap_sum*self.radius
            kap_dif_R = kap_dif*self.radius
            res =  -kap_dif**2 * (-1)**((1+lf+li)/2) * (kap_sum_R*np.cos(kap_sum_R) - np.sin(kap_sum_R))
            res += kap_sum**2 * (-1)**((1+li-lf)/2) * (kap_dif_R*np.cos(kap_dif_R) - np.sin(kap_dif_R))
            res /= 2.0 * kapi * kapf * kap_sum**2 * kap_dif**2
        return res

    # radial intergral for decay in dipole approximation
    # dimension GeV^-4
    # cache the results to save time
    # added approximate result at large kappa R
    def rad_int_B(self,ni,li,nf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        if abs(li-lf) != 1:
            raise ValueError('Calculating amplitude for unallowed transition')
        try:
            res = self.rad_int_B_cache[(ni,li,nf,lf)]
        except KeyError:
            res = self.rad_int(lambda r : self.RB(r,ni,li),self.kapB(ni,li),li,lambda r: self.RB(r,nf,lf),self.kapB(nf,lf),lf,force_full,subinterval_periods,approx_threshold)
            self.rad_int_B_cache[(ni,li,nf,lf)] = res
        return res

    # radial intergral for scattering in dipole approximation
    # dimension GeV^-4
    rad_int_S_cache = {}
    def rad_int_S(self,li,nf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        if abs(li-lf) != 1:
            raise ValueError('Calculating amplitude for unallowed transition')
        try:
            res = self.rad_int_S_cache[(li,nf,lf)]
        except KeyError:
            res = self.rad_int(lambda r : self.RS(r,li),self.kapS,li,lambda r: self.RB(r,nf,lf),self.kapB(nf,lf),lf,force_full,subinterval_periods,approx_threshold)
            self.rad_int_S_cache[(li,nf,lf)] = res
        return res

    # Triple product of spherical harmonics, integrated
    # This combination appears in the angular integrals
    def sph_prod(self,li,mi,mr,lf,mf):
        if abs(li-lf) != 1 or mi+mr != mf:
            return 0.0
        coef = np.sqrt(3.0/(4.0 * np.pi * (2*lf+1) * (2*li+1)))
        clres = 0.
        if lf == li + 1:
            clres = np.sqrt(comb(lf-mf,li-mi)*comb(lf+mf,li+mi))
        elif lf == li - 1:
            clres = (-1)**mr * np.sqrt(comb(li-mi,lf-mf)*comb(li+mi,lf+mf))
        else:
            raise ValueError('Unphysical spherical harmonic product')
        return coef * clres

    # Angular integral assembled for all three components
    def ang_int(self,li,mi,lf,mf):
        if abs(li-lf) != 1 or abs(mi-mf) > 1:
            return 0.0
        sph_x = self.sph_prod(li,mi,-1,lf,mf)-self.sph_prod(li,mi,1,lf,mf)
        sph_y = 1j*(self.sph_prod(li,mi,-1,lf,mf)+self.sph_prod(li,mi,1,lf,mf))
        sph_z = np.sqrt(2.0) * self.sph_prod(li,mi,0,lf,mf)
        return np.sqrt(2.0*np.pi/3.0) * np.array([sph_x,sph_y,sph_z])

    # amplitude
    # dimesionless
    def amp_B(self,ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        return self.Z * self.e * self.q(ni,li,nf,lf) * self.NB(ni,li) * self.NB(nf,lf) * self.rad_int_B(ni,li,nf,lf,force_full,subinterval_periods,approx_threshold) * self.ang_int(li,mi,lf,mf)

    # amplitude for scattering
    # in GeV^-3/2
    def amp_S(self,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        res = np.array([0.,0.,0.],dtype='complex128')
        for li in [lf-1,lf+1]:
            if li < 0 or self.k*self.radius < li:
                continue
            res += self.Z * self.e * (self.EB(nf,lf) + self.k**2/(2.0 * self.mu)) * self.NS(li) * self.NB(nf,lf) * np.sqrt((2.0*li+1)/(4.0*np.pi)) * self.rad_int_S(li,nf,lf,force_full,subinterval_periods,approx_threshold) * self.ang_int(li,0,lf,mf)
        return res

    def pol_tensor_full(self,ctq,phiq):
        ctq2 = ctq**2
        stq2 = 1.0 - ctq2
        stq = np.sqrt(stq2)
        cpq = np.cos(phiq)
        cpq2 = cpq**2
        spq2 = 1.0-cpq2
        spq = np.sqrt(spq2)
        if phiq > np.pi:
            spq = - spq
        return np.array([[ ctq2*cpq2 + spq2, -stq2*cpq*spq,    -ctq*stq*cpq ],
                         [ -stq2*cpq*spq,    cpq2 + ctq2*spq2, -ctq*stq*spq ],
                         [ -ctq*stq*cpq,     -ctq*stq*spq,     stq2]])

    def pol_tensor_phi_int_part(self,ctq,phiq):
        ctq2 = ctq**2
        stq2 = 1.0 - ctq2
        stq = np.sqrt(stq2)
        cpq = np.cos(phiq)
        cpq2 = cpq**2
        spq2 = 1.0-cpq2
        spq = np.sqrt(spq2)
        if phiq > np.pi:
            spq = - spq
        return np.array([[ 0.5 * (phiq * (1.0 + ctq2) - cpq * spq * stq2), -0.5 * stq2*spq2,                               -ctq*stq*spq ],
                         [ -0.5 * stq2 * spq2,                             0.5 * (phiq * (1.0 + ctq2) + cpq * spq * stq2), -ctq*stq*(1.0-cpq) ],
                         [ -ctq*stq*spq,                                   -ctq*stq*(1.0-cpq),                             phiq * stq**2]])

    def pol_tensor_ct_int_part(self,ctq,phiq):
        ctq2 = ctq**2
        stq2 = 1.0 - ctq2
        stq = np.sqrt(stq2)
        cpq = np.cos(phiq)
        cpq2 = cpq**2
        spq2 = 1.0-cpq2
        spq = np.sqrt(spq2)
        if phiq > np.pi:
            spq = - spq
        return (np.pi/3.0) * np.diag([4.0 + 3.0 * ctq + ctq**3, 4.0 + 3.0 * ctq + ctq**3, 2.0*(2.0-ctq)*(1.0+ctq)**2])


    def pol_tensor_phi_int(self,ctq):
        ctq2 = ctq**2
        stq2 = 1.0 - ctq2
        return np.pi * np.diag([ 1.0 + ctq2, 1.0 + ctq2, 2.0*stq2 ])
        
    #Double check this to ensure correctness (Jacob)
    def pol_tensor_ct_int(self,phiq):
        spq = np.sin(phiq)
        spq2 = spq**2
        cpq = np.cos(phiq)
        cpq2 = cpq**2
        return np.array[[2/3*cpq2, 4/3*cpq*spq, 0],
                        [4/3*cpq*spq, 2/3*spq2, 0],
                        [0, 0, 4/3]]

    pol_tensor_int = 8.0 * np.pi / 3.0

    ################################################################################
    # MAIN DECAY FUNCTIONS
    ################################################################################

    # decay rate differential in cos(theta) and phi of the photon (ctq,phiq) relative to spin z axis, GeV
    # eps: polarization of outgoing photon (+- 1 for right/left)
    # ni,li,mi: initial state quantum numbers
    # nf,lf,mf: final state quantum numbers
    def dGamma_B_dphidct(self,ctq,phiq,ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        amp = self.amp_B(ni,li,mi,nf,lf,mf,force_full,subinterval_periods,approx_threshold)
        return np.real(self.q(ni,li,nf,lf) * np.linalg.multi_dot([np.conjugate(amp),self.pol_tensor_full(ctq,phiq),amp]) / (8.0 * np.pi**2))

    # decay rate integrated in cos(theta) of the photon (ctq,phiq) relative to spin z axis, GeV
    # eps: polarization of outgoing photon (+- 1 for right/left)
    # ni,li,mi: initial state quantum numbers
    # nf,lf,mf: final state quantum numbers
    def dGamma_B_dct(self,ctq,ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        amp = self.amp_B(ni,li,mi,nf,lf,mf,force_full,subinterval_periods,approx_threshold)
        return np.real(self.q(ni,li,nf,lf) * np.linalg.multi_dot([np.conjugate(amp),self.pol_tensor_phi_int(ctq),amp]) / (8.0 * np.pi**2))


    # total decay rate in GeV
    # Independent of polarization (emerges as phase)
    # ni,li,mi: initial state quantum numbers
    # nf,lf,mf: final state quantum numbers
    def Gamma_B(self,ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        amp = self.amp_B(ni,li,mi,nf,lf,mf,force_full,subinterval_periods,approx_threshold)
        return np.real(self.q(ni,li,nf,lf) * self.pol_tensor_int * np.linalg.multi_dot([np.conjugate(amp),amp]) / (8.0 * np.pi**2)) # decay rate to all allowed states in GeV

    # decay rate to all allowed states in GeV
    # n,l,m: quantum numbers of decaying state
    # Returns: all allowed n,l,m final states with their respective decay rate
    '''parallelize TODO'''




    def Gamma_tot_B(self,n,l,m,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        res = {}
        #task_list = []
        for lf in [l-1,l+1]:
            if lf < 0:
                continue
            nf = self.nmax(lf,self.EB(n,l))
            while nf > 0 and self.q(n,l,nf,lf)*self.radius < np.pi:
                #changed from m-1 to m+2 since python doesn't account for the m+1 if stopping there, only reaches m.
                for mf in range(m-1,m+2):
                    if mf > lf or mf < -lf:
                        continue
                    #parallelizing in the future
                    #args.append((ni,li,mi,lf,nf,lf,force_full,subinterval_periods,approx_threshold))
                    rate = self.Gamma_B(n,l,m,nf,lf,mf,force_full,subinterval_periods,approx_threshold)
                    res[(nf,lf,mf)] = rate
                nf -= 1
                #with mp.Pool(mp.cpu_count()) as pool:
                    #rates = pool.map(self.Gamma_B, args)
                    #res[(nf,lf,mf)] = rates #somehow, not sure yet.
        return res

    def pdf_phi_B(self,phiq,ctq,ni,li,mi,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        amp = self.amp_B(ni,li,mi,nf,lf,mf,force_full,subinterval_periods,approx_threshold)


    ################################################################################
    # MAIN SCATTERING FUNCTIONS
    ################################################################################

    # scattering rate differential in cos(theta),phi for the photon (ctq) relative to spin z axis, GeV^-2
    def dxsec_v_S_dphidct(self,ctq,phiq,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        amp = self.amp_S(nf,lf,mf,force_full,subinterval_periods,approx_threshold)
        return np.real(self.EB(nf,lf) * np.linalg.multi_dot([np.conjugate(amp),self.pol_tensor_full(ctq,phiq),amp]) / (8.0 * np.pi**2))

    # scattering rate differential in cos(theta),phi for the photon (ctq) relative to spin z axis, GeV^-2
    def dxsec_v_S_dct(self,ctq,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        amp = self.amp_S(nf,lf,mf,force_full,subinterval_periods,approx_threshold)
        return np.real(self.EB(nf,lf) * np.linalg.multi_dot([np.conjugate(amp),self.pol_tensor_phi_int(ctq),amp]) / (8.0 * np.pi**2))

    # total cross section to given final state in GeV^-2
    def xsec_v_S(self,nf,lf,mf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        amp = self.amp_S(nf,lf,mf,force_full,subinterval_periods,approx_threshold)
        return np.real(self.EB(nf,lf) * self.pol_tensor_int * np.linalg.multi_dot([np.conjugate(amp),amp]) / (8.0 * np.pi**2))

    # cross section to all allowed states in GeV^-2
    # Possible revisions of xsec_v_tot_S will be written as comments within the function. Assuming that total should include n < nmax along with other quantum numbers for corrections 
    def xsec_v_tot_S(self,force_full = False, subinterval_periods = 8.0, approx_threshold = 10.0):
        res = {}
        for lf in range(int(np.ceil(self.k*self.radius)) + 1):
            nf = self.nmax(lf,0.) 
            #new code. accounts for nf<nmax that satisfies conditions. higher cross-section for many values of R.
            while nf > 0 and (self.EB(nf,lf) + self.k**2/(2*self.mu))*self.radius < np.pi:
                for mf in range(-1,2):
                    if mf > lf or mf < -lf:
                        continue
                    xsec_v = self.xsec_v_S(nf,lf,mf,force_full,subinterval_periods,approx_threshold)
                    res[(nf,lf,mf)] = xsec_v
                nf -= 1
        return res

    ################################################################################
    # SAMPLE cos(theta)
    ################################################################################

    def sample_ctq_phiq(self,ni,li,mi,nf,lf,mf):
            y = np.random.random()
            if mi == mf:
                delta = np.exp(1j * np.pi / 3.0) * (-1.0 + 2.0*y + 2.0 * 1j * np.sqrt((1.0-y)*y))**(1.0/3.0)
                phiq = self.sample_B_phiq(ni,li,mi,nf,lf,mf,(1.0/delta + delta))
                return (1.0/delta + delta), phiq
            elif abs(mi - mf) == 1:
                delta = (-2.0 + 4.0*y + np.sqrt(5.0 - 16.0*(1.0-y)*y))**(1.0/3.0) 
                phiq = self.sample_B_phiq(ni,li,mi,nf,lf,mf,(-1.0/delta + delta))
                return (-1.0/delta + delta), phiq
            raise ValueError('Transition not allowed')

    def sample_S_ctq_phiq(self,nf,lf,mf):
        selection = False
        while selection == False:
            ctq = np.random.uniform(-1,1)
            phiq = np.random.uniform(0,2*np.pi)
            #Normalization constant of probability distribution function
            Npdf = 1/self.xsec_v_S(nf,lf,mf)
            Mpdf = np.random.uniform(0,1)
            #Check height of probability distribution at ctq and phiq
            hd = Npdf*self.dxsec_v_S_dphidct(ctq,phiq,nf,lf,mf)
            if Mpdf<hd:
                selection = True
                break
        return(ctq,phiq)

    ####Temporary phiq rejection sampling, use till we have proper inverse transform sampling method for phiq
    def sample_B_phiq(self,ni,li,mi,nf,lf,mf,ctq):
        selection = False
        while selection == False:
            phiq = np.random.uniform(0,2*np.pi)
            #Normalization constant of probability distribution function
            Npdf = 1/self.Gamma_B(ni,li,mi,nf,lf,mf)
            Mpdf = np.random.uniform(0,1)
            #Check height of probability distribution at ctq and phiq
            hd = Npdf*self.dGamma_B_dphidct(ctq,phiq,ni,li,mi,nf,lf,mf)
            if Mpdf<hd:
                selection = True
                break
        return(phiq)

    ''' use if rejection turns out to be faster after limiting Mpdf to maximum pdf height
    def sample_B_ctq_phiq(self,ni,li,mi,nf,lf,mf):
        selection = False
        while selection == False:
            ctq = np.random.uniform(-1,1)
            phiq = np.random.uniform(0,2*np.pi)
            #Normalization constant of probability distribution function
            Npdf = 1/self.Gamma_B(ni,li,mi,nf,lf,mf)
            Mpdf = np.random.uniform(0,1)
            #Check height of probability distribution at ctq and phiq
            hd = Npdf*self.dGamma_B_dphidct(ctq,phiq,ni,li,mi,nf,lf,mf)
            if Mpdf<hd:
                selection = True
                break
        return(ctq,phiq)
    '''
