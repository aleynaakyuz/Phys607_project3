# This script is for reading the simulated
# parameters and creating frequency series data.
from pycbc.waveform import get_fd_waveform

def create_data(data):
    """
    Fuction for creating gravitational wave
    signal in frequeny domain. 

    Parameters:
    -----------
    data : h5py file, parameters that are read from h5 file.

    Returns:
    --------
    hp: pycbc Frequency series, plus phase in frequency domain.
    hc: pycbc Frequency series, cross phase in frequency domain.
    inp: dictonary, parameters used to create data.
    """
    
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

    data_dic = {key: data[dataset_key][:] for key, dataset_key in parameters.items()}
    inp = {**data_dic, **parameters2}
    hp, hc = get_fd_waveform(**inp)
    
    return hp, hc, inp