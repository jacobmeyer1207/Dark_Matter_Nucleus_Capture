# Imports************************************************************
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import time
import pandas as pd
import pyhepmc as hp
import DMNC_Detector as dmnc_det     # Comes with plt, np, rand, time
import DMNC_Rates as rts   # Access to many fundamental calculations
import pip
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import traceback
import cProfile
import pstats

# Functions**********************************************************

def sum_dict_vals(dictionary):
    '''Finds and returns the sum of the passed dictionary's values.
    
       Returns: the sum of a dictionary's stored data'''
    
    total = 0
    for key in dictionary.keys():
        total += dictionary[key]
    return total


def key_val_by_weight(dictionary):
    '''Selects a key-value pair from the dictionary by weight.
       The weighting is determined from the values, not keys.
    
       Returns: Tuple containing a selected key and value'''
    
    keys = list(dictionary.keys())
    total = sum_dict_vals(dictionary)
    weight = rand.uniform(0, total)
    
    # Define outside loop so they're in the correct namespace
    count = 0
    curr_key = 0
    curr_val = 0
    for i in range(len(keys)):
        curr_key = keys[i]
        curr_val = dictionary[curr_key]
        count += curr_val
        if count >= weight:
            return (curr_key, curr_val)
    # Temporary statement for testing purposes
    print('ERROR: "key_val_by_weight()" Exited the loop somehow')
    return (curr_key, curr_val)
    

def format_seconds(seconds):
    '''Takes a number of seconds and converts it to
       hour : minute : second format.
       
       Returns: string of hours:minutes:seconds'''
    
    hours = int(seconds // 3600)
    minutes = int(seconds // 60) % 60
    seconds = seconds % 60
    return f'{hours}:{minutes}:{seconds:.3f}'

# Parameters*********************************************************
face_count = {'front': 0,
              'back': 0,
              'right': 0,
              'left': 0,
              'top': 0,
              'bottom': 0}

det_length = 62                        # meters
det_width = 15.1                       # meters
det_height = 14                        # meters


# Number density of Liquid Argon, cm^-3:
num_density_LAr = 1.39 * 6.02e23 / 39.948

#####################################################################################################################################
################################################ Simulation Generation Functions ####################################################
#####################################################################################################################################

def Gen_DM_particle_event():
    start = time.monotonic()
    while True:
        energies = []
        try:
            det = dmnc_det.Detector(det_length, det_width, det_height, num_density_LAr)
            v = det.random_entrance()
            rates = rts.Rates(velDM = v)
            #setting cross-sections here to allow for random velocity implementation
            x_sec_dict = rates.xsec_v_tot_S() # keys: states, vals: X sects
            xsec_tot = sum_dict_vals(x_sec_dict)   # Total cross section, GeV^-2
            # hc is 1240 eV * nm: e-9 for eV -> GeV, e-7 for nm -> cm
            cm_from_inv_gev = 1240 / (2 * np.pi) * 1e-16
            xsec_cm = xsec_tot * cm_from_inv_gev**2  # Total cross section, cm^2
            det.set_xsec(x_sec_dict, xsec_cm)
            det.gen_capture_locs()
            event, photon_list = det.photon_generation() 
            for i in photon_list:
                energies.append(i[1])
            duration = time.monotonic() - start
            print('time to generate event:', format_seconds(duration))
            return(event, energies)
        except Exception:
            traceback.print_exc()
            raise



def process_one_event():

    event, energies = Gen_DM_particle_event()

    data = graph_data(event)

    photons = len(event.particles)

    captures = len(event.vertices)

    return data, photons, captures, energies


def Capture_stats(n = 1):
    data_long = []
    photon_long = 0
    vertecies_long = 0
    energies_long = []
    '''
    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_one_event, range(n)):
            data_long.extend(result[0])
            photon_long += result[1]
            vertecies_long += result[2]
            energies_long += result [3]
    '''
    for i in range(n):
        result = process_one_event()
        data_long.extend(result[0])
        photon_long += result[1]
        vertecies_long += result[2]
        energies_long += result [3]
    
    return data_long, photon_long, vertecies_long, energies_long

def print_event_summary(event):
    #print(event)
    print(f"Event {event.event_number}")
    print(f"Particles: {len(event.particles)}")
    print(f"Vertices : {len(event.vertices)}")
    print(f"Momentum units: {event.momentum_unit}")
    print(f"Length units  : CM")
    


def graph_data(event):
    particle_e_list = []
    for particle in event.particles:
        if particle.pid == 22:
            particle_e_list.append(particle.momentum.e)
    return(particle_e_list)




#####################################################################################################################################
############################################################# Various Tests #########################################################
#####################################################################################################################################
'''
####Testing photon energy generation sampling#########################
#plot in terms of n and l the photon energies
def plot_energies(n, rates):
    print(n)
    photon_energy_diff_tot = 0
    c = 0
    if(n > 1):
        T_data_n_l = []
        l_list = []
        phot_e = rates.EB(n,n-1)
        for i in range(n-2):
            curr_T_phot_e = rates.EB(n,i)
            if curr_T_phot_e > 0:
                c+=1
                phot_energy_dif = phot_e - curr_T_phot_e
                T_data_n_l.append(phot_energy_dif)
                l_list.append(i)
                photon_energy_diff_tot += phot_energy_dif
        print(len(T_data_n_l))
        plt.plot(l_list ,T_data_n_l, '.')
        plt.xlabel("E(l+1) - E(l) values (different n is different color)")
        plt.ylabel("Binding energy")
        if c != 0:
            print("average photon energy transition (only l transitions):",photon_energy_diff_tot/c)
        else:
            print("no valid states to go to")
        plot_energies(n-1, rates)
    else:
        return("done")
'''
'''
def children(n, l, m, rates):
    cross_sections = rates.Gamma_tot_B(n,l,m)
    states = cross_sections.keys()
    tot_cross_section = sum_dict_vals(cross_sections)
    for i in states:
        yield(i, cross_sections[i],tot_cross_section)


def all_transitions(rates):
    n = rates.nmax(1,0)
    queue = deque([(n,1,0)])
    visited = {(n,1,0)}

    photon_energies = []

    while queue:

        parent = queue.popleft()
        np, lp, mp = parent
        E_parent = rates.EB(np, lp)
        for child in children(np, lp, mp, rates):
            state, cross_section, tot_cross_section = child
            nc, lc, mc = state
            E_child = rates.EB(nc, lc)
            for i in range(int((cross_section/tot_cross_section)*100)):
                photon_energies.append(
                    (parent, state, E_parent - E_child)
            )
            if state not in visited:
                visited.add(state)
                queue.append(state)
        print("l,n,m visited:", parent)
    return photon_energies
'''

#Testing for proper rates.xsec_v_tot_s() functionality, plots vs R
#search for rad_int_s functionality next, implement into one_search below.


def one_search(r):
    rates = rts.Rates(R = r)
    x_sec_dict = rates.xsec_v_tot_S()
    xsec_tot = sum_dict_vals(x_sec_dict)
    if len(x_sec_dict) != 0:
        print("cross-section:",xsec_tot)
    else:
        print("FAILRUE: NO AVAILABLE TRANSITIONS")
        raise
    return x_sec_dict, xsec_tot

def search_all():
    searches = [np.arange(1,12,.001)]
    for j in searches:
        with ProcessPoolExecutor() as executor:
            max_cross_section = 0
            max_state_store = ()
            sec_list = []
            sec_len_list = []
            n_list = []
            l_list = []
            m_list = []
            for R, result in zip(j, executor.map(one_search, j)):
                rates = rts.Rates(R = R)
                x_sec_dict, xsec_tot = result
                max_state = max(x_sec_dict, key=x_sec_dict.get)
                n,l,m = max_state
                if max_cross_section < x_sec_dict[max_state]:
                    max_cross_section = x_sec_dict[max_state]
                    max_state_store = max_state
                sec_list.append(xsec_tot)
                sec_len_list.append(len(x_sec_dict))
                n_list.append(n)
                l_list.append(l)
                m_list.append(m)
            maxS = max(sec_list)
            print("max cross section:",maxS)
            print("max state:", max_state_store, "cross-section:", max_cross_section)
            print("associated R value:",R)
            print("average cross section:", sum(sec_list)/len(sec_list))
            n,l,m = max_state_store
            print('max state n value:',n,'l value:',l, 'm value:',m)
            plt.plot(j,sec_list,'.')
            plt.yscale('log')
            plt.xlabel('R value')
            plt.ylabel('cross-section total')
            plt.title(f"Cross-section VS R from {min(j)} - {max(j)}")
            plt.show()
            plt.plot(sec_len_list, sec_list,'.')
            plt.yscale('log')
            plt.xlabel('length of cross section list')
            plt.ylabel('cross-section total')
            plt.show()
            plt.plot(j,n_list,'.')
            plt.title('largest cross section contribution n value v.s. R')
            plt.xlabel('R value')
            plt.ylabel('state n value')
            plt.show()
            plt.plot(j,l_list,'.')
            plt.title('largest cross section contribution l value v.s. R')
            plt.xlabel('R value')
            plt.ylabel('state l value')
            plt.show()
            plt.plot(j,m_list,'.')
            plt.title('largest cross section contribution m value v.s. R')
            plt.xlabel('R value')
            plt.ylabel('state m value')
            plt.show()
            plt.plot(j,sec_len_list, '.')
            plt.title('number of states contributing to the overall cross section V.S. R')
            plt.xlabel('R value')
            plt.ylabel('length of contributing cross-sections list')
            plt.show()
           

####Testing methods of sampling angles################################
#do simple histogram test
'''
det = dmnc_det.Detector(det_length, det_width, det_height, num_density_LAr, xsec_cm, x_sec_dict)
det.gen_capture_locs()
Rejection_samples = []
Inversion_samples = []
key = list(det.capture_locs.keys())[0]
state = det.capture_locs[key]
ni = state[0]
li = state[1]
mi = state[2]
decay_dict = rates.Gamma_tot_B(ni, li, mi)
new_state = key_val_by_weight(decay_dict)[0]
nf = new_state[0]
lf = new_state[1]
mf = new_state[2]
cos_inv = 0
cos_rej = 0
for i in range(5000):
    cos_inv = rates.sample_ctq_phi(mi,mf)[0]
    cos_rej = rates.sample_B_ctq_phiq(ni,li,mi,nf,lf,mf)[0]
    Rejection_samples.append(cos_rej)
    Inversion_samples.append(cos_inv)
plt.hist(Inversion_samples, alpha=0.5, label='Inversion sampling', bins = 65)

plt.hist(Rejection_samples, alpha=0.5, label='Rejection sampling', bins = 65)

plt.legend(loc='upper right')
plt.title('Inverse transform sampling vs Rejection sampling')
plt.show()
'''

'''
for i in range(500000):
    face_count[det.random_face()] += 1

plt.title("Histogram of Entrance Location")
plt.grid(True)
plt.xlim(0, 7)
plt.bar(x=[i for i in range(1, 7)], height=[face_count[key] for key in face_count.keys()], tick_label = list(face_count.keys()))
plt.show()



for i in range(1000):
    det.random_entrance()
    if not det.particle_in_det():
        print('Test failed: particle outside detector')
        break
    unit_norm = np.sqrt(det.ux**2 + det.uy**2 + det.uz**2)
    if unit_norm < 0.999999999 or unit_norm > 1.000000001:
        print('Test failed: unit vector not normalized.')
        print('norm =', unit_norm)
        break
    if i == 99:
        print('Success! All tests passed')
'''

'''
####Testing radial wavefunction for proper amplitudes, l = 1
rates = rts.Rates()
def plot_wavefunc(n):
    for j in range(n):
        NB = rates.NB(j+1,1)
        R_list = []
        r_list = []
        for i in range(2000):
            r = 12*(i+1)/2001
            r_list.append(r)
            R = rates.RB(r,j+1,1) * NB
            R_list.append(R)
        plt.plot(r_list, R_list, label=f"n={j+1}")
        plt.legend()
    plt.xlabel('r')
    plt.ylabel('R_nl')
    plt.title("All radial wave functions as a function of r for the p-wave bound states with l = 1")
    plt.show()

n = rates.nmax(1,0)
plot_wavefunc(n)
'''
#####################################################################################################################################
############################################################# Main ##################################################################
#####################################################################################################################################
def main():

    ########### Testing block ##############
    search_all()
    '''
    E = []
    rates = rts.Rates()
    photon_energies = all_transitions(rates)
    for i in photon_energies:
        E.append(i[2])
    plt.hist(E, bins = int(np.sqrt(len(E))))
    plt.xlabel('photon energies')
    plt.ylabel('number of photons')
    plt.show()
    '''

    ########## Simulation block ############
    '''
    profiler = cProfile.Profile()

    profiler.enable()

    n = 1
    print(pip.__version__) 
    start_time = time.monotonic()
    particle_e_list, photon_num, capture_num, energies_list = Capture_stats(n)
    end_time = time.monotonic()
    duration = end_time - start_time
    print('Time to generate all events:', format_seconds(duration))
    print('Photons per second:', photon_num/duration)
    print('seconds per capture:', duration/capture_num)
    print('Total photons:', photon_num)
    plt.hist(particle_e_list, bins = int(np.sqrt(len(particle_e_list))))
    plt.xlabel("Photon Energy")
    plt.ylabel("Number of Photons")
    plt.title(f"{n} event(s). Photon energies produced:")
    plt.show()

    profiler.disable()

    stats = pstats.Stats(profiler)

    stats.sort_stats("cumtime").print_stats(30)
    '''



if __name__ == "__main__":
    main()
        




