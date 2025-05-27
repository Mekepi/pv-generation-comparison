import numpy as np
from os.path import dirname, abspath, isfile
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool
import pickle

from usine_functions import state_usines_pv_mmd_generation, usine_plot

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def usines_plot(state_Z_list:list[tuple[str, np.ndarray]]) -> None:
    with Pool(cpu_count()) as p:
        p.starmap(usine_plot, state_Z_list)

def usines_pv_mmd_generation() -> dict[str, np.ndarray]:
    if (isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'))):
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'), 'rb', 8*1024*1024) as fin:
            return pickle.load(fin)

    with Pool(cpu_count()) as p:
        result:list[tuple] = p.map(state_usines_pv_mmd_generation, list(states.values()))

    per_state_usines_pv_mmd_generation:dict[str, np.ndarray] = {state:array for state, array in result}

    with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'), 'wb', 8*1024*1024) as fout:
        pickle.dump(per_state_usines_pv_mmd_generation, fout)

    return per_state_usines_pv_mmd_generation