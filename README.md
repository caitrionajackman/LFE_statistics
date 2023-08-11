# Saturn LFE Statistics

*Insert short description of the project and aims here*

## Data
*Describes how to download and structure the data*

The LFE detection output of the UNET code (O'Dwyer et al. (YYYY)) can be found here:
* [UNET Output](https://zenodo.org/record/8075625)

The training data used to develop this model can be found and processed with the following:
* [Training Data (json)](https://zenodo.org/record/7895766)
* [Selection_of_Low_Frequency_Extensions_of_Saturn_Kilometric_Radiation](https://github.com/elodwyer1/Selection_of_Low_Frequency_Extensions_of_Saturn_Kilometric_Radiation/)

The list of planetary phase oscillations (PPO) used:
* [PPO Phases 2004-2017](https://figshare.le.ac.uk/articles/dataset/PPO_phases_2004-2017/10201442)

Cassini Plotting Repository:
* [Cassini_Plotting](https://zenodo.org/record/7349921)

## SPICE
If you're familiar with using SPICE and spiceypy, and have your own metakernel. You can skip to the end of this section and replace the path with a path to your metakernel.

To use spiceypy to quickly retrieve Cassini ephemeris we must donwload the relevant SPICE kernels. [DIAS SPICE Tools](https://github.com/mjrutala/DIASPICETools) (Rutala, M. J.) is a great package for easily downloading the required kernels without needing any experience with SPICE. It can be downloaded using the following command:

`git clone https://github.com/mjrutala/DIASPICETools`

Then open a python terminal:

run `python`

*inside python terminal*
run:
`from make_Metakernel import *`
`make_Metakernel("Cassini")`

This will download the relevant kernel files needed (~3 GB), this may take some time.

## Programs
*Describes what each file does and how to use the tools*

## Requirements
*Note to add version numbers*
* spiceypy
* matplotlib
* numpy


## References
O'Dwyer, E. P., Jackman, C.M, Domijan, K., & Lamy, L. (2023). Image-based Classification of Intense Radio Bursts from Spectrograms: An Application to Saturn Kilometric Radiation (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.807562

O'Dwyer, Elizabeth P., Jackman, Caitriona M., Domijan, Katarina, Lamy, Laurent, & Louis, Corentin K. (2023). Selection of Low Frequency Extensions of Saturn Kilometric Radiation detected by Cassini/RPWS. (2.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7895766

Provan, Gabrielle (2018). PPO phases 2004-2017. University of Leicester. Dataset. https://hdl.handle.net/2381/42436

Elizabeth O'Dwyer. (2022). elodwyer1/Cassini_Plotting: v1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.7349921