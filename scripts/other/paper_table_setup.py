from astropy.io import ascii
from astropy.table import Table
import pandas as pd
import yaml
from functions.file_handling import paths

def make_table(galaxy_name, loc):
    DIR_dict = paths(galaxy_name, loc)

    PNe_df = pd.read_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")

    index_check = PNe_df["ID"].isin(["-", "CrssMtch"])
    y_idx = PNe_df.loc[~index_check].index.values
    

    RA_for_table = [RA.replace("h", "").replace("m", "").replace("s", "") for RA in PNe_df["Ra (J2000)"].loc[y_idx]]  
    DEC_for_table = [DEC.replace("d", "").replace("m", "").replace("s", "") for DEC in PNe_df["Dec (J2000)"].loc[y_idx]]

    ID_for_table = ["F3D J"+RA_for_table[i]+DEC_for_table[i] for i in range(len(y_idx))]


    PNe_table = Table([ID_for_table, PNe_df["Ra (J2000)"].loc[~index_check], PNe_df["Dec (J2000)"].loc[~index_check],
                    PNe_df["m 5007"].loc[~index_check].round(2),
                    PNe_df["A/rN"].loc[~index_check].round(1),
                    PNe_df["PNe_LOS_V"].loc[~index_check].round(1),
                    PNe_df["ID"].loc[~index_check]],
                    names=("PN ID", "Ra", "Dec", "m 5007", "A/rN", "LOSVD", "ID"))


    # Save table in tab separated format.
    ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results.txt", format="tab", overwrite=True) 
    # Save latex table of data.
    ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results_latex.txt", format="latex", overwrite=True)

with open("config/galaxy_info.yaml", "r") as yaml_data:
    yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

for gal_loc in yaml_info:
    gal, loc = gal_loc.split("_")
    make_table(gal, loc)