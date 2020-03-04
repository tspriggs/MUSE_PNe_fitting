from astropy.io import fits, ascii
from astropy.table import Table
import pandas as pd
import argparse
import re
from functions.file_handling import paths

my_parser = argparse.ArgumentParser()


my_parser.add_argument('--galaxy', action='store', type=str, required=True)
my_parser.add_argument("--loc",    action="store", type=str, required=True)
args = my_parser.parse_args()

galaxy_name = args.galaxy 
loc = args.loc  

DIR_dict = paths(galaxy_name, loc)

PNe_df = pd.read_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")

y_idx = PNe_df.loc[PNe_df["ID"]!="-"].index.values

RA_for_table = [RA.replace("h", "").replace("m", "").replace("s", "") for RA in PNe_df["Ra (J2000)"].loc[PNe_df["ID"]!="-"]]  
DEC_for_table = [DEC.replace("d", "").replace("m", "").replace("s", "") for DEC in PNe_df["Dec (J2000)"].loc[PNe_df["ID"]!="-"]]

ID_for_table = ["F3D J"+RA_for_table[i]+DEC_for_table[i] for i in range(len(y_idx))]

PNe_table = Table([ID_for_table, PNe_df["Ra (J2000)"].loc[PNe_df["ID"]!="-"], PNe_df["Dec (J2000)"].loc[PNe_df["ID"]!="-"],
                   PNe_df["m 5007"].loc[PNe_df["ID"]!="-"].round(2),
                   PNe_df["A/rN"].loc[PNe_df["ID"]!="-"].round(1),
                   PNe_df["PNe_LOS_V"].loc[PNe_df["ID"]!="-"].round(1),
                   PNe_df["ID"].loc[PNe_df["ID"]!="-"]],
                   names=("PN ID", "Ra", "Dec", "m 5007", "A/rN", "LOSVD", "ID"))


# Save table in tab separated format.
ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results.txt", format="tab", overwrite=True) 
# Save latex table of data.
ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results_latex.txt", format="latex", overwrite=True) 