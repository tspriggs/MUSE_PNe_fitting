import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

galaxy_df = pd.read_csv("exported_data/galaxy_dataframe.csv")


plt.figure(figsize=(20,8))
plt.scatter(galaxy_df["Galaxy"], galaxy_df["dM PNLF"], label="PNLF", c="r")
plt.scatter(galaxy_df["Galaxy"], galaxy_df["lit dM"], label="Literature (SBF)", c="k", alpha=0.7)

plt.errorbar(galaxy_df["Galaxy"], galaxy_df["dM PNLF"], yerr=galaxy_df["dM PNLF err"], ls="None", c="r" ,lw=1.1)
plt.errorbar(galaxy_df["Galaxy"], galaxy_df["lit dM"], yerr=galaxy_df["lit dM err"], ls="None", c="k", alpha=0.7)

plt.axhline(galaxy_df["dM PNLF"].loc[galaxy_df["PNe N"]>20].median(), ls="--", alpha=0.6, label="Median PNLF D")
plt.axhline(galaxy_df["lit dM"].median(), c="k", ls="--", alpha=0.6, label="Median SBF D")

plt.legend(loc="upper right", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.ylabel("Distance modulus", fontsize=15)
plt.xlabel("Fornax Galaxy Name", fontsize=15)

# plt.fill_between(x=galaxy_df["Galaxy"], y1=galaxy_df["lit dM"].max(), y2=galaxy_df["lit dM"].min(), color="k", alpha=0.1)
# plt.fill_between(x=galaxy_df["Galaxy"], y1=galaxy_df["dM PNLF"].max(), y2=galaxy_df["dM PNLF"].min(), color="r", alpha=0.1)
plt.savefig("Plots/Distance_comp_lit.png", bbox_inches='tight')