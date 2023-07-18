import pandas as pd

input_data_fp = "./../data/"

spiceTrajectoriesDetections = pd.read_csv(input_data_fp + '/lfe_detections.csv',parse_dates=['start','end'])

bethTrajectoriesDetections = pd.read_csv(input_data_fp + "/closest_xyz_startlfe.csv",parse_dates=["start"])

for (i, spiceLine), (j, bethLine)in zip(spiceTrajectoriesDetections.iterrows(), bethTrajectoriesDetections.iterrows()):
    print(spiceLine["x_ksm"] - bethLine["xpos_ksm"])
