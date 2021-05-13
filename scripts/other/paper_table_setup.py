from astropy.io import ascii
from astropy.table import Table
import numpy as np
import pandas as pd
import yaml
from functions.file_handling import paths


def make_table(galaxy_name, loc):
    DIR_dict = paths(galaxy_name, loc)

    PNe_df = pd.read_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")

    index_check = PNe_df["ID"].isin(["-"])
    y_idx = PNe_df.loc[~index_check].index.values
    
    RA_for_table = [RA.replace("h", "").replace("m", "").replace("s", "") for RA in PNe_df["Ra (J2000)"].loc[y_idx]]  
    DEC_for_table = [DEC.replace("d", "").replace("m", "").replace("s", "") for DEC in PNe_df["Dec (J2000)"].loc[y_idx]]

    ID_for_table = ["F3D J"+RA_for_table[i]+DEC_for_table[i] for i in range(len(y_idx))]
    m_5007 = PNe_df["m 5007"].loc[~index_check].round(2).values
    m_5007_err = PNe_df.loc[~index_check, ["mag error up", "mag error lo"]].median(1).round(2).values

    PNe_LOSV = PNe_df["PNe_LOS_V"].loc[~index_check].round(1).values
    PNe_LOSV_err = PNe_df["PNe_LOS_V_err"].loc[~index_check].round(1).values


    PNe_table = Table([ID_for_table, PNe_df["Ra (J2000)"].loc[~index_check], PNe_df["Dec (J2000)"].loc[~index_check],
                    m_5007, m_5007_err, 
                    PNe_df["A/rN"].loc[~index_check].round(1),
                    PNe_LOSV, PNe_LOSV_err,
                    PNe_df["ID"].loc[~index_check], PNe_df["index"].loc[~index_check]],
                    names=("PN ID", "Ra", "Dec", "m 5007", "mag err", "A/rN", "LOSV","LOSV err", "ID", "index"))


    # Save table in tab separated format.
    ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results.txt", format="tab", overwrite=True) 
    # Save latex table of data.
    ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results_latex.txt", format="latex", overwrite=True)

with open("config/galaxy_info.yaml", "r") as yaml_data:
    yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

for gal_loc in yaml_info:
    gal, loc = gal_loc.split("_")

    if gal not in ["FCCtest","FCC090", "FCC263", "FCC285", "FCC290", "FCC306", "FCC308", "FCC312"]:
        make_table(gal, loc)