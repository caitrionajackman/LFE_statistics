import pandas as pd
import spiceypy as spice
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

saturnRadius = 60268 # km
spice.furnsh("./SPICE/cassini/metakernel_cassini.txt")
spice.furnsh("./SPICE/cassini/cas_dyn_v03_Daragh.tf")

input_data_fp = "./../data/"

print("Loading trajectories")

bethTrajectoriesData = pd.read_csv(input_data_fp + "cassini_output/trajectorytotal.csv")

print("Reading data")

step = 0.25 # days

times = pd.to_datetime(bethTrajectoriesData["datetime_ut"], format="%Y-%m-%d %H:%M:%S")[0::int(1440*step)] # 1 day / n steps
x = bethTrajectoriesData["xpos_ksm"][0::int(1440*step)]
y = bethTrajectoriesData["ypos_ksm"][0::int(1440*step)]
z = bethTrajectoriesData["zpos_ksm"][0::int(1440*step)]

print("Passing dates to spice")

ets = spice.datetime2et(times)

print("Pulling from spice")
positions, ltimes = spice.spkpos("Cassini", ets, "CASSINI_KSM", "NONE", "SATURN")

print("Dividing by Rs")
positions_Rs = []
for position in tqdm(positions, total=len(positions)):
    position_Rs = np.divide(position, saturnRadius)

    positions_Rs.append(position_Rs)

positions_Rs = np.array(positions_Rs)
spiceX = positions_Rs[:,0]
spiceY = positions_Rs[:,1]
spiceZ = positions_Rs[:,2]

print("Plotting")
fig, axes = plt.subplots(1, 2)
xy_axis, xz_axis = axes

xy_axis.plot(spiceX, spiceY, color="indianred", label="SPICE")
xz_axis.plot(spiceX, spiceZ, color="indianred", label="SPICE")

xy_axis.plot(x, y, color="mediumturquoise", label="trajectorytotal.csv (PDS Data)")
xz_axis.plot(x, z, color="mediumturquoise", label="trajectorytotal.csv (PDS Data)")

xy_axis.set_xlabel("X$_{KSM}$ (R$_S$)")
xy_axis.set_ylabel("Y$_{KSM}$ (R$_S$)")

xz_axis.set_xlabel("X$_{KSM}$ (R$_S$)")
xz_axis.set_ylabel("Z$_{KSM}$ (R$_S$)")

for axis in axes:
    axis.set_aspect("equal")
    axis.legend()

plt.show()
