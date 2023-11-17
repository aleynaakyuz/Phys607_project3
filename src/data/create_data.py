import h5py
from pycbc.waveform import get_fd_waveform

parameters = {
    "mass1": "mass1",
    "mass2": "mass2",
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "coa_phase": "coa_phase",
    "inclination": "inclination",
    "distance": "distance",
    "ra": "ra",
    "dec": "dec",
    "polarization": "polarization"
}

parameters2 = {
    "approximant": "IMRPhenomXPHM",
    "delta_f": 0.1,
    "f_lower": 10
}

data = h5py.File('parameter.h5', 'r')
data_dic = {key: data[dataset_key][:] for key, dataset_key in parameters.items()}

inp = {**data_dic, **parameters2}

hp, hc = get_fd_waveform(**inp)

hf = h5py.File('data.h5', 'w')
hf.create_dataset('hp', data=hp)
hf.create_dataset('hc', data=hc)
hf.close()