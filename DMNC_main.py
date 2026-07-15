# Imports************************************************************
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import time
import pandas as pd
import pyhepmc as hp

import pip
print(pip.__version__) 



import DMNC_Detector as dmnc_det     # Comes with plt, np, rand, time


import DMNC_Rates as rts   # Access to many fundamental calculations
rates = rts.Rates()
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

########################################################################## LOOK INTO THIS #TODO ###############################
x_sec_dict = rates.xsec_v_tot_S() # keys: states, vals: X sects
xsec_tot = sum_dict_vals(x_sec_dict)   # Total cross section, GeV^-2

# hc is 1240 eV * nm: e-9 for eV -> GeV, e-7 for nm -> cm
cm_from_inv_gev = 1240 / (2 * np.pi) * 1e-16

xsec_cm = xsec_tot * cm_from_inv_gev**2  # Total cross section, cm^2

###############################################################################################################################

# Number density of Liquid Argon, cm^-3:
num_density_LAr = 1.39 * 6.02e23 / 39.948

# Main***************************************************************

def Gen_DM_particle_event():
    i = 0
    while True:
        i += 1
        try:
            det = dmnc_det.Detector(det_length, det_width, det_height, num_density_LAr, xsec_cm, x_sec_dict)
            det.random_entrance()
            det.gen_capture_locs()
            event = det.photon_generation()
            print("Captures:", len(det.capture_locs))
            print("number of failed captures:", i)
            print("Cross-section of capture in cm:", xsec_cm)
            return(event)
        except ValueError as ex:
            continue

def Capture_stats():
    start_time = time.monotonic()
    event = Gen_DM_particle_event()
    end_time = time.monotonic()
    duration = end_time - start_time
    print('Time to generate all decays:', format_seconds(duration))
    capture_num = len(event.vertices)
    graph_data(event)
    print_event_summary(event)

def print_event_summary(event):
    print(event) 
    ''' for if the event is large
    print(f"Event {event.event_number}")
    print(f"Particles: {len(event.particles)}")
    print(f"Vertices : {len(event.vertices)}")
    print(f"Weights  : {event.weights}")
    print(f"Momentum units: {event.momentum_unit}")
    print(f"Length units  : {event.length_unit}")
    '''


def graph_data(event):
    particle_e_list = []
    for particle in event.particles:
        if particle.pid == 22:
            particle_e_list.append(particle.momentum.e)
    plt.hist(particle_e_list, bins = int(np.sqrt(len(particle_e_list))))
    plt.xlabel("Photon Energy")
    plt.ylabel("Number of Photons")
    plt.show()

Capture_stats()


####Testing photon energy generation sampling#########################
#plot in terms of n and l the photon energies
'''
def plot_energies(n, T_phot_e):
    if(n > 1):
        T_data_n_l_trans = []
        for i in range(n-1):
            curr_T_phot_e = rates.EB(n,i)
            T_phot_diff = T_phot_e - curr_T_phot_e
            T_data_n_l_trans.append(T_phot_diff)
            T_phot_e = curr_T_phot_e
        print(len(T_data_n_l_trans))
        plt.plot(T_data_n_l_trans, range(n-1), '.')
        T_phot_e = rates.EB(n-1,0)
        plot_energies(n-1, T_phot_e)
    else:
        return("done")
T_phot_e = rates.EB(85,0)
print(plot_energies(85, T_phot_e))
plt.show()
'''




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
#Testing for proper rates.xsec_v_tot_s() functionality, plots vs R
'''
searches = [np.arange(10, 12, .001)]
for j in searches:
    print("beginning task")
    i_list = []
    sec_list = []
    sec_len_list = []
    for i in j:
        rates = rts.Rates(R = i)
        x_sec_dict = rates.xsec_v_tot_S() # keys: states, vals: X sects
        xsec_tot = sum_dict_vals(x_sec_dict)   # Total cross section, GeV^-2
        if len(x_sec_dict) != 0:
            largest_state = max(x_sec_dict.items(), key=lambda x: x[1])

        print(
                "radius:",rates.radius,
                #"largest state:",largest_state[0],     # (n,l,m)
                #"cross-section:",largest_state[1],     # cross section
                "total cross-section",xsec_tot,
                #"fraction of total:",largest_state[1]/xsec_tot,
            )
        if len(x_sec_dict) != 0:
            print("largest state:",largest_state[0],     # (n,l,m)
            "cross-section:",largest_state[1],
            "fraction of total:",largest_state[1]/xsec_tot)
            print("average cross-section:",xsec_tot/len(x_sec_dict))
        else:
            print("ERROR: NO CROSS-SECTIONS FOUND.")
        sec_len_list.append(len(x_sec_dict))
        i_list.append(i)
        sec_list.append(xsec_tot)
    maxS = max(sec_list)
    print("max cross section:",maxS)
    print("associated R value:",i_list[sec_list.index(maxS)])
    print("average cross section:", sum(sec_list)/len(sec_list))
    plt.plot(i_list,sec_list,'.')
    plt.yscale('log')
    plt.xlabel('R value')
    plt.ylabel('cross-section total')
    plt.show()
    plt.plot(sec_len_list, sec_list,'.')
    plt.yscale('log')
    plt.xlabel('length of cross section list')
    plt.ylabel('cross-section total')
    plt.show()
    '''



