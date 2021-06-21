import re
import os
import pickle
import logging
import pprint

from ase import io, Atoms
from collections import defaultdict, Counter

logging.basicConfig(filename="output_extra.log",
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


class CatalystResult:
    def __init__(self, catalyst_name, reaction_dict):
        """

        :param catalyst_name: Name/label of the catalyst we want to collate data for
        This class is used to handle the various outputs in a computational chemistry catalyst simulation.
               reaction_dict: Dictionary mapping an intermediate to the relevant things which distinguish it from
               the vacancy intermediate. ie.
               label2info = {
                                "VAC": [[], 0, 0],
                                "OH": [[8, 1], 0, oh_energy],
                                "OOH": [[8, 8, 1], 0, ooh_energy],
                            }
                Where the first value in the list is a list of integer values for the atomic numbers in a given intermediate,
                the second value is the charge difference between VAC and a specific intermediate, and the third value
                is the floating point value which denotes the value we need to calculate the binding energies based
                on the standard hydrogen electrode model.
        """
        self.cat_name = catalyst_name
        self.intermediates_sorted = defaultdict(list)
        self.intermediates = []
        self.binding_energies = defaultdict(float)
        self.ele_binding_energies = defaultdict(float)

        self.ground_state_G = defaultdict(float)
        # Use this list so that we don't attempt to calculate binding energies for incomplete/failed calculations
        self.incomplete_intermediates = []
        for key in reaction_dict:
            self.ground_state_G[key] = 0
            self.intermediates_sorted[key] = []
            if key != "VAC":
                self.binding_energies[key] = 0

    def add_intermediate(self, intermediate_instance):
        self.intermediates.append(intermediate_instance)

    def sort_intermediates(self):
        """
        Sort the intermediates into their respective intermediate names, supplied by reaction_dict
        :return: Nothing
        """
        len_minimum = 1000
        for intermediate in self.intermediates:
            if len(intermediate.structure.get_atomic_numbers()) < len_minimum:
                len_minimum = len(intermediate.structure.get_atomic_numbers())
                vac_counter = Counter(intermediate.structure.get_atomic_numbers())
                charge = intermediate.charge
        for intermediate in self.intermediates:
            for intermediate_label in reaction_dict:
                intermediate_counter = Counter(intermediate.structure.get_atomic_numbers())
                difference_between_vac = intermediate_counter - vac_counter
                diff_atoms = []
                for key in difference_between_vac:
                    for i in range(difference_between_vac[key]):
                        diff_atoms.append(key)
                if Counter(diff_atoms) == Counter(reaction_dict[intermediate_label][0]) and\
                        intermediate.charge == charge+reaction_dict[intermediate_label][1]:
                    self.intermediates_sorted[intermediate_label].append(intermediate)
        logging.info("Intermediates sorted for: {}".format(self.cat_name))

    def get_ground_states(self):
        """
        Sort the intermediates on the basis of their energy, where we require the ground state ie. minimum
        :return:
        """
        for intermediate_key in self.intermediates_sorted:
            if intermediate_key not in self.incomplete_intermediates:
                for intermediate in self.intermediates_sorted[intermediate_key]:
                    if self.ground_state_G[intermediate_key] == 0:
                        self.ground_state_G[intermediate_key] = intermediate

                    if intermediate.gibbs_energy < self.ground_state_G[intermediate_key].gibbs_energy:
                        self.ground_state_G[intermediate_key] = intermediate
                        logging.info("Found a potential ground state for: {} ".format(self.cat_name+intermediate_key))

    def set_binding_energies(self):
        """

        :return: Nothing
        Calculates the binding energies according to the t
        """
        assert self.intermediates_sorted["VAC"]!=0
        for intermediate_key in self.intermediates_sorted:
            if self.intermediates_sorted[intermediate_key]==[] or intermediate_key=="VAC":
                # no complete logfile for this intermediate
                continue
            else:
                self.binding_energies[intermediate_key] = (self.ground_state_G[intermediate_key].gibbs_energy - \
                                                          self.ground_state_G["VAC"].gibbs_energy - \
                                                           reaction_dict[intermediate_key][2])*27.2114
                self.ele_binding_energies[intermediate_key] = (self.ground_state_G[intermediate_key].electronic_energy - \
                                                          self.ground_state_G["VAC"].electronic_energy - \
                                                           reaction_dict[intermediate_key][2])*27.2114

    def run(self, list_of_logs):
        """
        Handles
        :param list_of_logs: the list of log files pertaining to a specific catalyst.
        :return: the CatalystResult object, with the updates gained from analysing outputs.
        """
        checks = []
        for log in list_of_logs:
            intermediate_to_add = Intermediate(log)
            self.add_intermediate(intermediate_to_add)
            checks.append(intermediate_to_add.check_minimum())
        if False in checks:
            return self
        self.sort_intermediates()

        self.get_ground_states()
        self.set_binding_energies()
        return self


class Intermediate:
    def __init__(self, logfile):
        logging.info("Reading: {}".format(logfile))
        self.logfile = logfile
        self.gibbs_energy = None
        self.electronic_energy = None
        self.multiplicity = None
        self.charge = None
        self.status = 'unknown'

        with open(self.logfile, "r") as f:
            self.lines = f.readlines()

        for idx, line in enumerate(self.lines[:1000]):
            if "Charge =" in line:
                self.charge = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                self.multiplicity = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[1])
                logging.info("Charge and multiplicity are {} and {} for {}".format(self.charge, self.multiplicity,
                                                                                   self.logfile))
            if "NAtoms=" in line:
                self.num_atoms = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

        for idx, line in enumerate(self.lines):
            if "Standard orientation" in line:
                self.lines_w_xyz = self.lines[idx+5:idx+5+self.num_atoms]

        xyz = []
        atomic_nums = []
        for over_line in self.lines_w_xyz:
            xyz.append([float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", over_line)[-3:]])
            atomic_nums.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", over_line)[1]))

        self.structure = Atoms(atomic_nums, positions=xyz)

    def check_minimum(self):
        """
        :return: Boolean telling us whether the simulation was successful.
        """
        converged_run = False
        if "Normal termination" in self.lines[-1]:
            converged_run = True

        if converged_run:
            # Check whether imaginary
            for line in self.lines[-20000:]:
                if "Frequencies" in line:
                    smallest_freq = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                    if smallest_freq < 0:
                        self.status = "imaginary_freq"
                        logging.info("Imaginary frequency for {}".format(self.logfile))
                        return False
                    else:
                        self.status = "true_minimum"
                        self.get_g_and_e()
                        return True

        else:
            # Check slurm, check how it broke.
            pass
            logging.info("Fail for: {}".format(self.logfile))
            return False

    def get_g_and_e(self):
        """
        Set the gibbs free energy and electronic energy.
        :return: Nothing
        """
        for line in self.lines[-10000:]:
            if "Thermal correction to Gibbs Free Energy=" in line:
                gibbs_correction = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                self.correction = gibbs_correction
            elif "Sum of electronic and thermal Free Energies=" in line:
                self.gibbs_energy = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                # To get E without dispersion correction for solv correction.
                self.electronic_energy = self.gibbs_energy - gibbs_correction
                logging.info("Free energy for {} is {}".format(self.logfile, self.gibbs_energy))
                break


# Define this path wherever the catalyst data is stored, must be organised as
# path_to_catalysts/catalyst_name/intermediate_name/...
# the string definition also must contain the forward slash as the final character.
path_to_catalysts = "/home/michael/new_scal/organised_molsimps/"
logfiles = os.popen("find {} -type f -name '*.log'".format(path_to_catalysts)).readlines()

logfiles = [x[:-1] for x in logfiles]

catalyst2logfiles = defaultdict(list)
cat_name_directory_level = path_to_catalysts.count("/")
intermediate_name_directory_level = cat_name_directory_level+1
for log in logfiles:
    catalyst2logfiles[log.split("/")[cat_name_directory_level]].append(log)

# Define basis sets of intermediate energies, changed depending on function, example provided below for water splitting
# units are atomic units (Ha)

h2_energy = -1.176286
h2o_energy = -76.439515

# Use above as basis set to define binding energies.
oh_energy = h2o_energy - 0.5*h2_energy
o_energy = h2o_energy - h2_energy
ooh_energy = 2*h2o_energy - 1.5*h2_energy

label2info = {
    "VAC": [[], 0, 0],
    "OH": [[8, 1], 0, oh_energy],
    "oh4": [[8, 1], 1, oh_energy + 4.28 / 27.2114],
    "oxo": [[8], 0, o_energy],
    "oxo5": [[8], 1, o_energy + 4.28 / 27.2114],
    "OOH": [[8, 8, 1], 0, ooh_energy],
    "ooh4": [[8, 8, 1], 1, ooh_energy + 4.28 / 27.2114]
}

finished_logs = []
cat2results = defaultdict()
binding_energies = defaultdict()
ele_binding_energies = defaultdict()

for cat_name in catalyst2logfiles:
    print(cat_name)
    intermediate_names = []
    for log in catalyst2logfiles[cat_name]:
        intermediate_name = log.split("/")[intermediate_name_directory_level]
        assert intermediate_name in label2info.keys()
        intermediate_names.append(intermediate_name)
    intermediate_names = list(set(intermediate_names))

    cat_res = CatalystResult(cat_name, label2info).run(catalyst2logfiles[cat_name])
    binding_energies[cat_name] = cat_res.binding_energies
    ele_binding_energies[cat_name] = cat_res.ele_binding_energies
    for label in intermediate_names:
        if cat_res.binding_energies[label]!=0:
            finished_logs.append(cat_res.ground_state_G[label].logfile)
    print(cat_res.binding_energies)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(cat2results)

pickle.dump(binding_energies, open("gibbs_binding_energies.p", "wb"), protocol=2)
