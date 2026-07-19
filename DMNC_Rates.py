################################################################################
# ENVIRONMENT
################################################################################

import scipy as sp

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
# global cache generation
from diskcache import Cache

cache = Cache('./DMNC_RATES_CACHE')
#cache.clear()
# Derivatives of spherical bessel functions
def spherical_jnp(l,x):
        return -0.5 * spherical_jn(l,x) / x + np.sqrt(0.5 * np.pi / x) * jvp(l+0.5,x)
def spherical_ynp(l,x):
    return -0.5 * spherical_yn(l,x) / x + np.sqrt(0.5 * np.pi / x) * yvp(l+0.5,x)
# Spherical bessel function zeros as floats (no mpmath)
def spherical_jnz(l,n):
    return float(besseljzero(l+0.5,n))

@cache.memoize()
def EB_cache(n,l,Z,A,e,mu,V0,radius):
    res = V0 - 0.5 * kapB_cache(n,l,Z,A,e,mu,V0,radius)**2 / mu
    #if res > 0.:
        #levels[(n,l,radius)] = res
    return res

@cache.memoize()
def kapB_cache(n,l,Z,A,e,mu,V0,radius):
    res = spherical_jnz(l,n)/radius
    return res

@cache.memoize()
def q_cache(ni,li,nf,lf,Z,A,e,mu,V0,radius):
    E_ph = - EB_cache(ni,li,Z,A,e,mu,V0,radius) + EB_cache(nf,lf,Z,A,e,mu,V0,radius)
    if E_ph <= 0.:
        raise ValueError('Attempting transition from lower energy state to higher energy state')
    return E_ph

@cache.memoize()
def nmax_cache(l,Emin,Z,A,e,mu,V0,radius):
    res = int(np.ceil((np.sqrt(2.0*mu*V0)*radius)/np.pi - 0.5 * l))
    while res > 0 and EB_cache(res,l,Z,A,e,mu,V0,radius) < Emin:
        res -= 1
    while res + 1 > 0 and EB_cache(res+1,l,Z,A,e,mu,V0,radius) > Emin:
        res += 1
    return res

@cache.memoize()
def NB_cache(n,l,Z,A,e,mu,V0,radius):    
    normint = -0.25*(np.pi*radius**3*jv(-0.5 + l,spherical_jnz(l,n))*jv(1.5 + l,spherical_jnz(l,n)))/spherical_jnz(l,n)
    return 1.0/np.sqrt(normint)

def RB(r,n,l,Z,A,e,mu,V0,radius):
    return spherical_jn(l,kapB_cache(n,l,Z,A,e,mu,V0,radius)*r)

@cache.memoize()
def rad_int_B_cache(ni,li,nf,lf,Z,A,e,mu,V0,radius,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    if abs(li-lf) != 1:
        raise ValueError('Calculating amplitude for unallowed transition')
    res = rad_int_cache(lambda r : RB(r,ni,li,Z,A,e,mu,V0,radius),kapB_cache(ni,li,Z,A,e,mu,V0,radius),li,lambda r: RB(r,nf,lf,Z,A,e,mu,V0,radius),kapB_cache(nf,lf,Z,A,e,mu,V0,radius),lf,Z,A,e,mu,V0,radius,force_full,subinterval_periods,approx_threshold)
    return res

@cache.memoize()
def sph_prod_cache(li,mi,mr,lf,mf,Z,A,e,mu,V0,radius):
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

@cache.memoize()
def amp_B_radial_cache(ni,li,nf,lf,Z,A,e,mu,V0,radius,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    return Z * e * q_cache(ni,li,nf,lf,Z,A,e,mu,V0,radius) * NB_cache(ni,li,Z,A,e,mu,V0,radius) * NB_cache(nf,lf,Z,A,e,mu,V0,radius) * rad_int_B_cache(ni,li,nf,lf,Z,A,e,mu,V0,radius,force_full,subinterval_periods,approx_threshold)

@cache.memoize()
def Gamma_B_cache(ni,li,mi,nf,lf,mf,pol_tensor_int,Z,A,e,mu,V0,radius):
    amp = amp_B_cache(ni,li,mi,nf,lf,mf,Z,A,e,mu,V0,radius)
    return np.real(q_cache(ni,li,nf,lf,Z,A,e,mu,V0,radius) * pol_tensor_int * np.vdot(amp, amp) / (8.0 * np.pi**2))

#############################FIXME: unfinished caches. functional for other cache calls, however don't cache values themselves. (errors from using memoize as method)
#@cache.memoize()
def amp_B_cache(ni,li,mi,nf,lf,mf,Z,A,e,mu,V0,radius):
    amp_r = amp_B_radial_cache(ni,li,nf,lf,Z,A,e,mu,V0,radius,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0)
    return amp_r * ang_int_cache(li,mi,lf,mf,Z,A,e,mu,V0,radius)

#@cache.memoize()
def ang_int_cache(li,mi,lf,mf,Z,A,e,mu,V0,radius):
    if abs(li-lf) != 1 or abs(mi-mf) > 1:
        return 0.0
    sph_x = sph_prod_cache(li,mi,-1,lf,mf,Z,A,e,mu,V0,radius)-sph_prod_cache(li,mi,1,lf,mf,Z,A,e,mu,V0,radius)
    sph_y = 1j*(sph_prod_cache(li,mi,-1,lf,mf,Z,A,e,mu,V0,radius)+sph_prod_cache(li,mi,1,lf,mf,Z,A,e,mu,V0,radius))
    sph_z = np.sqrt(2.0) * sph_prod_cache(li,mi,0,lf,mf,Z,A,e,mu,V0,radius)
    return np.sqrt(2.0*np.pi/3.0) * np.array([sph_x,sph_y,sph_z])

#@cache.memoize()
def rad_int_cache(Ri,kapi,li,Rf,kapf,lf,Z,A,e,mu,V0,radius,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
    if kapi*radius < approx_threshold or kapf*radius < approx_threshold or force_full:
        print('Calculating full radial integral... may be slow')
        subinterval_lim = max(50,int(np.ceil(max(kapi*radius,kapf*radius)/subinterval_periods)))
        rad_int_full = quad(lambda r : Ri(r)*Rf(r)*r**3,0,radius,limit=subinterval_lim)
        if rad_int_full[1] > 0.01 * abs(rad_int_full[0]):
            print('Error on radial integral greater than 1%')
        res = rad_int_full[0]
    else:
        kap_sum = kapi + kapf
        kap_dif = kapi - kapf
        kap_sum_R = kap_sum*radius
        kap_dif_R = kap_dif*radius
        res =  -kap_dif**2 * (-1)**((1+lf+li)/2) * (kap_sum_R*np.cos(kap_sum_R) - np.sin(kap_sum_R))
        res += kap_sum**2 * (-1)**((1+li-lf)/2) * (kap_dif_R*np.cos(kap_dif_R) - np.sin(kap_dif_R))
        res /= 2.0 * kapi * kapf * kap_sum**2 * kap_dif**2
    return res

# MARK: Class **************************************************************************
class Rates:
    def __init__(self, Z = 18, A = 40, mu_mult = .938, V0_mult = .246, R = 10.0, velDM = .001):
    ################################################################################
    # PARAMETERS
    ################################################################################
        self.Z = Z                                  # material atomic number
        self.A = A                                  # material mass number
        self.e = np.sqrt(4.0 * np.pi / 137)         # electric charge, dimensionless
        self.mu = A * mu_mult                       # nucleus (reduced) mass, GeV
        self.V0 = A * V0_mult                       # potential depth, GeV (A* expectation value of the higgs)
        self.k = velDM * self.mu                         # incoming DM momentum, GeV (.001 until initialized by set_rand_velocity. TODO: Fix DM lab frame momenta)
        self.radius = R                             # DM radius, GeV^-1
        # For States**************************************************************
        self.kapS = np.sqrt(self.k**2 + 2.0 * self.mu * self.V0)    # interior momentum for scattering, initialized in rand_vel, GeV
        self.pol_tensor_int = 8.0 * np.pi / 3.0
        # Caches******************************************************************
        self.levels = {}                            # cache for energy levels
        self.rad_int_B_cache = {}
        self.ang_int_cache = {}
        self.rad_int_S_cache = {}
        self.NB_cache = {}
        self.kapB_levels = {} 


    ################################################################################
    # STATES
    ################################################################################

    # momentum inside potential well for bound state, GeV
    def kapB(self,n,l):
        return kapB_cache(n,l,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

    # (positive) binding energy for state, GeV
    def EB(self,n,l):
        return EB_cache(n,l,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

    # emitted photon energy/momentum, GeV
    def q(self,ni,li,nf,lf):
        return q_cache(ni,li,nf,lf,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

    # Maximum n allowed to have bound states below top of potential
    def nmax(self,l,Emin):
        return nmax_cache(l,Emin,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

    # normalization for bound state, GeV^-3/2
    def NB(self,n,l):
        return NB_cache(n,l,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

    # boundary conditions for scattering state
    def bcs(self,Ns,delta,l):
        return ((np.cos(delta) * spherical_jn(l,self.k*self.radius) + np.sin(delta) * spherical_yn(l,self.k*self.radius)) - Ns * spherical_jn(l,self.kapS*self.radius),
                (np.cos(delta) * self.k * spherical_jnp(l,self.k*self.radius) + np.sin(delta) * self.k * spherical_ynp(l,self.k*self.radius)) - Ns * self.kapS * spherical_jnp(l,self.kapS*self.radius))
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
    def rad_int(self,Ri,kapi,li,Rf,kapf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0, cache = True):
        if cache == True:
            return rad_int_cache(Ri,kapi,li,Rf,kapf,lf,self.Z,self.A,self.e,self.mu,self.V0,self.radius,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0)
        else:
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
        return rad_int_B_cache(ni,li,nf,lf,self.Z,self.A,self.e,self.mu,self.V0,self.radius,force_full,subinterval_periods,approx_threshold)

    # radial intergral for scattering in dipole approximation
    # dimension GeV^-4
    def rad_int_S(self,li,nf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        if abs(li-lf) != 1:
            raise ValueError('Calculating amplitude for unallowed transition')
        try:
            res = self.rad_int_S_cache[(li,nf,lf)]
        except KeyError:
            res = self.rad_int(lambda r : self.RS(r,li),self.kapS,li,lambda r: self.RB(r,nf,lf),self.kapB(nf,lf),lf,force_full,subinterval_periods,approx_threshold,cache = False)
            self.rad_int_S_cache[(li,nf,lf)] = res
        return res

    # Triple product of spherical harmonics, integrated
    # This combination appears in the angular integrals
    def sph_prod(self,li,mi,mr,lf,mf):
        return sph_prod_cache(li,mi,mr,lf,mf,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

    # Angular integral assembled for all three components
    def ang_int(self,li,mi,lf,mf):
        return ang_int_cache(li,mi,lf,mf,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

    # amplitude
    # dimesionless
    def amp_B_radial(self,ni,li,nf,lf,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        return amp_B_radial_cache(ni,li,nf,lf,self.Z,self.A,self.e,self.mu,self.V0,self.radius,force_full,subinterval_periods,approx_threshold)

    def amp_B(self,ni,li,mi,nf,lf,mf):
        return amp_B_cache(ni,li,mi,nf,lf,mf,self.Z,self.A,self.e,self.mu,self.V0,self.radius)

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

    def pol_tensor_ct_int(self,phiq):
        spq = np.sin(phiq)
        spq2 = spq**2
        cpq = np.cos(phiq)
        cpq2 = cpq**2
        return np.array[[2/3*cpq2, 4/3*cpq*spq, 0],
                        [4/3*cpq*spq, 2/3*spq2, 0],
                        [0, 0, 4/3]]

    

    ################################################################################
    # MAIN DECAY FUNCTIONS
    ################################################################################

    # decay rate differential in cos(theta) and phi of the photon (ctq,phiq) relative to spin z axis, GeV
    # eps: polarization of outgoing photon (+- 1 for right/left)
    # ni,li,mi: initial state quantum numbers
    # nf,lf,mf: final state quantum numbers
    def dGamma_B_dphidct(self,ctq,phiq,ni,li,mi,nf,lf,mf):
        amp = self.amp_B(ni,li,mi,nf,lf,mf)
        return np.real(self.q(ni,li,nf,lf) * np.linalg.multi_dot([np.conjugate(amp),self.pol_tensor_full(ctq,phiq),amp]) / (8.0 * np.pi**2))

    # decay rate integrated in cos(theta) of the photon (ctq,phiq) relative to spin z axis, GeV
    # eps: polarization of outgoing photon (+- 1 for right/left)
    # ni,li,mi: initial state quantum numbers
    # nf,lf,mf: final state quantum numbers
    def dGamma_B_dct(self,ctq,ni,li,mi,nf,lf,mf):
        amp = self.amp_B(ni,li,mi,ni,lf,mf)
        return np.real(self.q(ni,li,nf,lf) * np.linalg.multi_dot([np.conjugate(amp),self.pol_tensor_phi_int(ctq),amp]) / (8.0 * np.pi**2))


    # total decay rate in GeV
    # Independent of polarization (emerges as phase)
    # ni,li,mi: initial state quantum numbers
    # nf,lf,mf: final state quantum numbers
    def Gamma_B(self,ni,li,mi,nf,lf,mf):
        return Gamma_B_cache(ni,li,mi,nf,lf,mf,self.pol_tensor_int,self.Z,self.A,self.e,self.mu,self.V0,self.radius) # decay rate to all allowed states in GeV

    # decay rate to all allowed states in GeV
    # n,l,m: quantum numbers of decaying state
    # Returns: all allowed n,l,m final states with their respective decay rate
    '''parallelize TODO'''


    #def worker(self,job):
    #    ni,li,mi,nf,lf,mf,force_full,subinterval_periods,approx_threshold = job
    #    amp_r = amp_B_radial(ni,li,nf,lf,force_full,subinterval_periods,approx_threshold)
    #    return self.Gamma_B(li,mi,lf,mf,amp_r)

    def Gamma_tot_B(self,n,l,m,force_full = False,subinterval_periods = 8.0,approx_threshold = 10.0):
        res = {}
        #job_list = []
        #states = []
        for lf in [l-1,l+1]:
            if lf < 0:
                continue
            nf = self.nmax(lf,self.EB(n,l))
            while nf > 0 and self.q(n,l,nf,lf)*self.radius < np.pi:
                #changed from m-1 to m+2 since python doesn't account for the m+1 if stopping there, only reaches m.
                for mf in range(m-1,m+2):
                    if mf > lf or mf < -lf:
                        continue
                    #parallelizing in the future (Use chunksize = larger number to increase efficiency)
                    #job_list.append((n,l,m,nf,lf,mf,force_full,subinterval_periods,approx_threshold))
                    res[nf,lf,mf] = self.Gamma_B(n,l,m,nf,lf,mf)
                nf -= 1
                #with ProcessPoolExecutor() as executor:
                #    res_results = executor.map(self.worker, job_list)

                #    res = dict(zip(states,res_results))
        return res

    


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

