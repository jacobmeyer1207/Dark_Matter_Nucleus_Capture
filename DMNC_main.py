# Imports************************************************************
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import time
import pandas as pd
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance

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

# FIXME: R value from Rates is used, not this one. I think I'll have
# to turn Rates into a class so I can modify values.
R = 10                                 # DM radius, GeV^-1
x_sec_dict = rates.xsec_v_tot_S()            # keys: states, vals: X sects
xsec_tot = sum_dict_vals(x_sec_dict)   # Total cross section, GeV^-2

# hc is 1240 eV * nm: e-9 for eV -> GeV, e-7 for nm -> cm
cm_from_inv_gev = 1240 / (2 * np.pi) * 1e-16

xsec_cm = xsec_tot * cm_from_inv_gev**2  # Total cross section, cm^2

# Number density of Liquid Argon, cm^-3:
num_density_LAr = 1.39 * 6.02e23 / 39.948

# Main***************************************************************

# Right now I am using this section for miscellaneous tests.
# I comment out the ones not in use, but save them because
# they are still useful for seeing how things are working.

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


def Capture_stats():
    det = dmnc_det.Detector(det_length, det_width, det_height, num_density_LAr, xsec_cm, x_sec_dict)
    avg_phot_energy_list = []
    decay_num_list = []
    det.random_entrance()
    det.gen_capture_locs()
    
    start_time = time.monotonic()

    det.photon_generation()
    end_time = time.monotonic()
    duration = end_time - start_time
    print('Time to generate all decays:', format_seconds(duration))
    capture_num = len(det.capture_locs.items())
    if capture_num > 0:
        for i in det.phot_4_vec_list:
            curr_avg_photon_energy = 0
            curr_decay_num = len(i)
            for j in i:
                curr_avg_photon_energy += j[0]
            curr_avg_photon_energy /= curr_decay_num
            avg_phot_energy_list.append(curr_avg_photon_energy)
            decay_num_list.append(curr_decay_num)
        return(det.phot_4_vec_list, capture_num, decay_num_list, duration, avg_phot_energy_list, det.capture_locs)
    else:
        print("failed a capture")
        return Capture_stats()


def HepMC_data():
    data = Capture_stats()
    capture_locs = list(data[5].keys())
    j = 1
    for i in data[0]:
        E = 0
        px = 0
        py = 0
        pz = 0
        print("Event:", j)
        print("Units: GeV, Meters")
        print("Vertex:", capture_locs[j-1]) #meters from the origin
        l = 1
        for k in i:
            print("photon ", l, ", 4-vector:", k)
            E += k[0]
            px += k[1]
            py += k[2]
            pz += k[3]
            l += 1
        avg_vec = [E/len(i), px/len(i), py/len(i), pz/len(i)]
        print("average photon 4-vector:", avg_vec)
        print("Capture of Argon atom produced", data[2][j-1],"photons.")
        j+=1

HepMC_data()
####Testing methods of sampling angles######################
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
for i in range(10):
    cos_inv += rates.sample_ctq_phi(mi,mf)[0]
    cos_rej += rates.sample_B_ctq_phiq(ni,li,mi,nf,lf,mf)[0]
cos_inv /= 10
cos_rej /= 10
print(cos_inv, cos_rej)
'''



