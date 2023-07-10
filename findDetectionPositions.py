import spiceypy as spice

from datetime import datetime
import pandas
import numpy as np

saturnRadius = 58232 # km

spice.furnsh("./SPICE/cassini/metakernel_cassini.txt")
spice.furnsh("./SPICE/cassini/cas_dyn_v03.tf")

lfeDetections = pandas.read_csv("./../data/lfe_timestamps.csv")
startTimes = lfeDetections["start"]
endTimes = lfeDetections["end"]
labels = lfeDetections["label"]



dates = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f") for time in startTimes]
ets = [spice.datetime2et(date) for date in dates]


positions, ltimes = spice.spkpos("Cassini", ets, "CASSINI_KSM", "NONE", "SATURN")

positions_Rs = []
for position in positions:
    position_Rs = np.divide(position, saturnRadius)

    positions_Rs.append(position_Rs)

newData = {
    "start": startTimes,
    "end": endTimes,
    "label": labels,
    "x_ksm": np.array(positions_Rs)[:,0],
    "y_ksm": np.array(positions_Rs)[:,1],
    "z_ksm": np.array(positions_Rs)[:,2],
}

df = pandas.DataFrame(newData)
df.to_csv("./../data/lfe_detections.csv")
