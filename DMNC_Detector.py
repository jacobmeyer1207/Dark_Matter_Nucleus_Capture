# Dark Matter Nucleus Capture Simulation
# 10/17/2024
# Jackson Benz, Professor Joshua Berger, Ethan Rutledge
# 
# This code randomly generates dark matter particles propagating through the DUNE far detector.
#      The inverse CDF method is used to randomly determine how many interactions occur between
#      the DM and the LAr in DUNE's FD. An interaction between the DM and LAr is then treated
#      as radiative nucleus capture, and the emitted photons from an Ar nucleus falling into
#      the DM's potential well are generated and tracked using Professor Berger's cross section-
#      calculating code.
#
# Updates:
# 10/17/2024: Added a graph to show the behavior of cross section.
#
# 10/24/2024: Time behavior of calculating cross section calculated and plotted.
#
# 11/07/2024: Begin simulation of the photon decay and nucleus capture as a whole.
#     Approximations in place:
#          Not boosting for the photon generation; I'm assuming that the difference between
#               center of momentum and lab frame is small and won't deflect photons much.
#          I am assuming the photons decay nearly instantly, so I do not need to propagate the
#               dark matter while decay occurs. This allows me to generate the locations of
#               interaction first and then generate photons afterward, which is convenient.
# 7/9/2026: Jacob Meyer now working on this code along with Joshua Berger and Zack Orr. Recent additions to the code include
#   the generation of momentum energy 4-vectors in photon generation and the switch to pyhepmc for storing event data. 
#   Main and Rates were changed to include caching for photon energies, rejection sampling for bound and scattering state captures,
#   a function generating various statistics from capture data, and basic plotting of photon energies generated for each capture. 
#   Currently testing Inverse transform sampling for cos(theta) angle of outgoing photons against rejection sampling method to ensure usability.




# Packages***********************************************************
from calendar import c
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import time
import DMNC_Rates as rts
import pyhepmc as hp

rates = rts.Rates()
# Class**************************************************************
class Detector:
    def __init__(self, length, width, height, num_density, cross_section_cm, cross_sec_dict):
        self.length = length
        self.width = width
        self.height = height
        
        # Origin is in the middle of the detector:
        self.x_min = -length / 2
        self.x_max = length / 2
        self.y_min = -width / 2
        self.y_max = width / 2
        self.z_min = -height / 2
        self.z_max = height / 2
        
        # Calculate area to randomly weight the entrance location (LW/LW+LH+WH gives probability of entering that face if each unit area has equal probability)
        self.area_x = height * width
        self.area_y = height * length
        self.area_z = width * length
        self.area_tot = self.area_x + self.area_y + self.area_z
        
        # Position tracking variables
        self.x = 0
        self.y = 0
        self.z = 0
        # Velocity
        self.vx = 0
        self.vy = 0
        self.vz = 0
        # Unit vector for direction (unit velocity)
        self.ux = 0
        self.uy = 0
        self.uz = 0
        self.time = 0
        
        # Dictionary with keys: capture location, values: state
        self.capture_locs = {}
        
        # Parameters important to scattering
        self.num_density = num_density
        self.xsec_cm = cross_section_cm
        self.xsec_dict = cross_sec_dict
        # Arbitrary PID number for HepMC formatting.
        self.QBALL_PID = 1000000001
        self.BOUND_QBALL_AR_PID = 1000000002


    def random_face(self):
        '''Randomly select which face DM will enter the detector
           through, using the dimensions the object was given.
           This method is mainly used for testing randomness, see
           "random_entrance()" for usable trajectories.
           
           Returns: string stating which face was entered.'''
        
        enter_face = rand.uniform(0, self.area_tot)
        
        if enter_face < self.area_x:     # Entering front or back face
            if rand.randint(0, 1) == 0:
                face = "front"  # +x
            else:
                face = "back"   # -x
        elif enter_face < self.area_x + self.area_y:   # Right or left
            if rand.randint(0, 1) == 0:
                face = "right"  # +y
            else:
                face = "left"   # -y
        elif enter_face <= self.area_tot:              # Top or bottom
            if rand.randint(0, 1) == 0:
                face = "top"    # +z
            else:
                face = "bottom" # -z
    
        return face
    
    
    def random_entrance(self):
        '''Randomly select an entrance point (x, y, z) on the detector,
           as well as the trajectory (ux, uy, uz) the DM takes through it.
           
           Returns: N/A.'''
        
        # Randomly select face to enter (using weighted areas)
        enter_face = rand.uniform(0, self.area_tot)
        # Randomly select trajectory using spherical coords
        cos = rand.uniform(-1, 1)
        phi = rand.uniform(0, 2 * np.pi)
        
        if enter_face < self.area_x:     # Entering front or back face
            if phi > np.pi / 2 and phi < 3 * np.pi / 2:
                # Front face
                self.x = self.x_max
                self.y = rand.uniform(self.y_min, self.y_max)
                self.z = rand.uniform(self.z_min, self.z_max)
            else:
                # Back face
                self.x = self.x_min
                self.y = rand.uniform(self.y_min, self.y_max)
                self.z = rand.uniform(self.z_min, self.z_max)
        elif enter_face < self.area_x + self.area_y:   # Right or left
            if phi > np.pi:
                # Right face
                self.x = rand.uniform(self.x_min, self.x_max)
                self.y = self.y_max
                self.z = rand.uniform(self.z_min, self.z_max)
            else:
                # Left face
                self.x = rand.uniform(self.x_min, self.x_max)
                self.y = self.y_min
                self.z = rand.uniform(self.z_min, self.z_max)
        elif enter_face <= self.area_tot:              # Top or bottom
            if cos < 0:
                # Top face
                self.x = rand.uniform(self.x_min, self.x_max)
                self.y = rand.uniform(self.y_min, self.y_max)
                self.z = self.z_max
            else:
                # Bottom face
                self.x = rand.uniform(self.x_min, self.x_max)
                self.y = rand.uniform(self.y_min, self.y_max)
                self.z = self.z_min
                
        # Position of entrance has been selected. Now use cos and phi
        # to make the unit vector of direction with ux, uy, and uz.
        sin = (1 - cos**2)**(0.5)
        self.ux = sin * np.cos(phi)
        self.uy = sin * np.sin(phi)
        self.uz = cos
    
    
    def particle_in_det(self):
        '''Boolean statement of whether particle is in the detector or not.
        
           Returns: True if particle is in detector, otherwise False'''
        return (self.x >= self.x_min and
                self.y >= self.y_min and
                self.z >= self.z_min and
                self.x <= self.x_max and
                self.y <= self.y_max and
                self.z <= self.z_max)
    
    
    def gen_capture_locs(self):
        '''Uses an inverse CDF to randomly generate where a nucleus
           is captured by the dark matter, then stores these locations.
           Stops once the DM leaves the detector.
           
           This method will call "random_entrance()"
           if it hasn't been called yet.
            
           Returns: N/A: stores locations in self.capture_locs
           
           #ASSUMES LOSS OF ENERGY AND MOMENTUM UPON CAPTURE IS NEGLIGIBLE#
           '''
        if (self.ux == 0 and self.uy == 0 and self.uz == 0):
            self.random_entrance()
        
        # Max number of iterations to avoid accidental infinite loop
        for i in range(10000):
            # random number between 0 and 1
            rand_num = rand.random()
            if rand_num == 1:
                break       # Because ln(0) -> infinity
            
            # Distance to next capture in cm
            cap_dist = ( -1 / (self.num_density * self.xsec_cm) *
                   np.log(1 - rand_num ))
            # Convert to meters:
            cap_dist *= 1e-2
            self.time += cap_dist/(rates.k/rates.mu)
            self.x += self.ux * cap_dist
            self.y += self.uy * cap_dist
            self.z += self.uz * cap_dist
            if self.particle_in_det():
                capture_state = key_val_by_weight(self.xsec_dict)[0]
                self.capture_locs[(self.x, self.y, self.z, self.time)] = capture_state
            else:
                break
    
    def photon_generation(self, event_num = 1):
        '''Simulates the decay of a nucleus in the DM potential well
           and stores the energies and momenta of the emitted photons.
           
           This function looks through the starting states stored in
           the keys of self.capture_locs
           
           Returns: pyhepmc.event'''
        ##################Seems to be causing problems on failed capture###################
        if len(self.capture_locs) == 0:
            # No decays to be done
            raise ValueError("failed capture")
        print("Capture successful, beginning photon generation.")
        ###################################################################################
        Argon_capture = 0 #describes the number of atoms containted within the Argon for PID selection purposes. 
        event = hp.GenEvent() #generation of particle event, may need to account for events of size zero in the future to see failed capture events. (TODO)
        event.event_number = event_num 
        ########################## FIXME ###############################
        a = hp.FourVector(0,0,0,0)
        curr_DM = hp.GenParticle(a,0,0) # initializing DM particle outside of loop for proper event structure
        ################################################################
        for location in self.capture_locs.keys():
            position = hp.FourVector(location[0], location[1], location[2], location[3])
            vertex = hp.GenVertex(position)
            q_state = self.capture_locs[location]
            n,l,m = q_state
            photon_energy = rates.EB(n, l)
            photon_ct, photon_phi = rates.sample_S_ctq_phiq(n,l,m)
            sin = (1 - photon_ct**2)**(0.5)
            DM_px = rates.k * self.ux
            DM_py = rates.k * self.uy
            DM_pz = rates.k * self.uz
            DM_e = np.sqrt(rates.k**2 + rates.mu**2)
            ######ADJUSTED PHOTON MOMENTUM TO BE RELATIVE TO THE REST FRAME RATHER THAN THE DM FRAME#################
            phot_px = photon_energy * (sin * np.cos(photon_phi) + DM_px/DM_e)
            phot_py = photon_energy * (sin * np.sin(photon_phi) + DM_py/DM_e)
            phot_pz = photon_energy * (photon_ct + DM_pz/DM_e)
            ### Non-relativistic, adjust if velocity is increased #######
            photon_energy = np.sqrt(phot_px**2 + phot_py**2 + phot_pz**2)
            #########################################################################################################
            momentum = hp.FourVector(phot_px, phot_py, phot_pz, photon_energy)
            particle = hp.GenParticle(momentum, 22, 1)
            vertex.add_particle_out(particle)
            momentum_Ar = hp.FourVector(0, 0, 0, rates.mu)
            particle_Ar = hp.GenParticle(momentum_Ar,1000180400,4)
            vertex.add_particle_in(particle_Ar)
            # momentum 4 vector of DM calculation
            momentum_DM = hp.FourVector(DM_px, DM_py, DM_pz, DM_e)
            if Argon_capture > 0:
                vertex.add_particle_in(curr_DM)
            else:
                curr_DM = hp.GenParticle(momentum_DM, self.QBALL_PID, 4)
                vertex.add_particle_in(curr_DM)
            # For now have a limited range just in case
            for i in range(1000):
                if n <= 1:
                    break
                decay_dict = rates.Gamma_tot_B(n, l, m)
                # Can be used to make sure we're below ns time
                total_decay_rate = sum_dict_vals(decay_dict)
                new_state = key_val_by_weight(decay_dict)[0]
                new_n, new_l, new_m = new_state
                
                photon_energy = rates.q(n, l, new_n, new_l)
                photon_ct, photon_phi = rates.sample_ctq_phi(m,new_m)
                photon_ct = photon_ct.real
                photon_phi = photon_phi.real
                sin = (1 - photon_ct**2)**(0.5)

                phot_px = photon_energy * (sin * np.cos(photon_phi) + DM_px/DM_e)
                phot_py = photon_energy * (sin * np.sin(photon_phi) + DM_py/DM_e)
                phot_pz = photon_energy * (photon_ct + DM_pz/DM_e)
                ### Non-relativistic, adjust if velocity is increased #######
                photon_energy = np.sqrt(phot_px**2 + phot_py**2 + phot_pz**2)

                momentum = hp.FourVector(phot_px, phot_py, phot_pz, photon_energy)
                particle = hp.GenParticle(momentum, 22, 1)
                vertex.add_particle_out(particle)
                
                # Update quantum numbers for next loop iteration
                n = new_n
                l = new_l
                m = new_m
                
                if i == 999:
                    print('ATTENTION: Did not finish decay. n =', n)
            Argon_capture += 1
            momentum_DM_Ar = hp.FourVector(DM_px,DM_py,DM_pz,DM_e) # NOTICE: ASSUMING PHOTON ENERGY AND MOMENTUM IS NEGLIGIBLE!
            next_DM = hp.GenParticle(momentum_DM_Ar, self.BOUND_QBALL_AR_PID, 1)
            vertex.add_particle_out(next_DM)
            next_DM = curr_DM
            event.add_vertex(vertex)
        return event
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
