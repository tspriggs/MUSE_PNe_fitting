import numpy as np
import pandas as pd
import yaml
import os
import argparse

from functions.file_handling import paths
from functions.completeness import prep_completness_data, calc_completeness

# Read in galaxy dataframe for list of galaxy names.
galaxy_df = pd.read_csv(f"exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))

# Setup of Argparse, where we can ask for individual galaxies by name, or, by default, all galaxies in galaxy_df will be used.
# dM type argument is for when the literature distance should be used instead.
# app argument is to switch to apperture summation, instead of FOV summation.
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', nargs="+", type=str, required=False, default=np.unique(galaxy_df.index.get_level_values(0)))

args = my_parser.parse_args()

galaxy_selection = args.galaxy 


gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
# loc = "center"

with open("config/galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

step = 0.001
m_5007 = np.arange(26, 31, step)

if os.path.isfile("exported_data/completeness_ratio_df.csv") is False:
    comp_df = pd.DataFrame(columns=("Galaxy", "FWHM", "beta", "LSF"))
    comp_df["Galaxy"] = np.unique(gal_df.index.get_level_values(0))
    comp_df.set_index("Galaxy", inplace=True)

else:
    comp_df = pd.read_csv("exported_data/completeness_ratio_df.csv", index_col="Galaxy")

for i, gal in enumerate(galaxy_selection):
    for loc in ["center", "halo", "middle"]:
        if f"{gal}_{loc}" in [*galaxy_info]:
            galaxy_data = galaxy_info[f"{gal}_{loc}"]
            print(f"Calculating {gal}'s {loc} completeness ratio....")
            DIR_dict = paths(gal, loc)
            image, Noise_map = prep_completness_data(gal, loc, DIR_dict, galaxy_data)

            completeness_ratio = calc_completeness(image, Noise_map, m_5007, galaxy_data, 3.0, 9, )

            np.save(DIR_dict["EXPORT_DIR"]+"_completeness_ratio", completeness_ratio)
        else:
            continue


