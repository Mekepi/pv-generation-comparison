import numpy as np
from time import perf_counter
from os import listdir
from os.path import dirname, abspath, isfile
from pathlib import Path
from scipy.spatial import cKDTree
from psutil import cpu_count
from multiprocessing import Pool
from gzip import open as gzopen
from os import makedirs
from collections import defaultdict
import pickle

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def city_process(data_folder:Path, ventures_folder:Path, state_timeseries_coords_folder:Path, state:str, city:str) -> tuple[str, defaultdict[str, defaultdict[float, list[float]]]]:

    with open("%s\\%s\\%s"%(ventures_folder, state, city), 'r', 8*1024*1024, encoding='utf-8') as file:
        ventures:np.ndarray = np.asarray(file.readlines()[1:], str)

    city_timeseries_coords_file:Path = Path("%s\\%s"%(state_timeseries_coords_folder, next(f for f in listdir(state_timeseries_coords_folder) if f.startswith(city[:9]))))
    city_timeseries_coords:np.ndarray = np.loadtxt(city_timeseries_coords_file, delimiter=',', ndmin=2, encoding='utf-8')

    failty_coord:list[str] = []
    city_ventures_coords_list:list[list[float]] = []
    for i in range(len(ventures)):
        venture_data:list[str] = ventures[i].split('";"')
        try: city_ventures_coords_list.append([float('.'.join(venture_data[3].split(','))), float('.'.join(venture_data[2].split(','))), float(i)])
        except Exception:
            failty_coord.append(ventures[i])
    city_ventures_coords:np.ndarray = np.asarray(city_ventures_coords_list)

    if failty_coord:
        with open("%s\\failty_coord.csv"%(data_folder), 'a', encoding='utf-8') as f:
            f.writelines(failty_coord)

    distances:list[float]
    idxs:list[int]
    distances, idxs = cKDTree(city_timeseries_coords).query(city_ventures_coords[:, :2], 1, workers=-1) # type: ignore

    faridxs:list[str] = ["(%7.2f,%6.2f) (%11.6f,%6.6f) %6.2f    %s"%(*city_ventures_coords[:, :2][i],*city_timeseries_coords[idxs[i]],distances[i],ventures[i]) for i in range(len(distances)) if distances[i]>=0.03]
    
    if faridxs:
        makedirs("%s\\outputs\\Too Far Coords\\%s"%(data_folder, state), exist_ok=True)
        with open("%s\\outputs\\Too Far Coords\\%s\\%s-too-far.csv"%(data_folder, state, city[:9]), 'w', 1024*1024*256, encoding='utf-8') as f:
            f.write("source coord;closest timeseries coord;distance;line\n")
            f.writelines(faridxs)
    
    coord_year_list:defaultdict[str, defaultdict[float, list[float]]] = defaultdict(defaultdict[float, list[float]])
    
    for venture, timeseries_coord in zip(ventures[city_ventures_coords[:,2].astype(int)], city_timeseries_coords[idxs]):
        venture_data = venture.split('";"')
        if ('(%.6f,%.6f)'%(timeseries_coord[1], timeseries_coord[0]) not in coord_year_list):
            coord_year_list['(%.6f,%.6f)'%(timeseries_coord[1], timeseries_coord[0])] = defaultdict(list[float])
        if (venture_data[27][:4] not in coord_year_list['(%.6f,%.6f)'%(timeseries_coord[1], timeseries_coord[0])]):
            coord_year_list['(%.6f,%.6f)'%(timeseries_coord[1], timeseries_coord[0])][float(venture_data[27][:4])] = [0., 0.]
        coord_year_list['(%.6f,%.6f)'%(timeseries_coord[1], timeseries_coord[0])][float(venture_data[27][:4])][0] += float('.'.join(venture_data[4].split(',')))
        coord_year_list['(%.6f,%.6f)'%(timeseries_coord[1], timeseries_coord[0])][float(venture_data[27][:4])][1] += 1
    
    return (city, coord_year_list)

def ventures_process(sts:list[str] = [], geocodes:list[str] = []) -> defaultdict[str, defaultdict[str, dict[str, np.ndarray]]]:

    if (isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'))):
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'rb', 32*1024*1024) as fin:
            return pickle.load(fin)

    data_folder:Path = Path(dirname(abspath(__file__))).parent

    ventures_folder:Path = Path('%s\\data\\ventures'%(data_folder))

    timeseries_coords_folder:Path = Path('%s\\data\\timeseries_coords'%(data_folder))

    states_irradiance:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = defaultdict(defaultdict[str, dict[str, np.ndarray]])

    with Pool(cpu_count()) as p:

        with open("%s\\failty_coord.csv"%(data_folder), 'w', encoding='utf-8') as f:
            f.close()

        for state in listdir(ventures_folder)[1:]:

            if ((sts and not(state[:2] in sts)) or (geocodes and not(state[:2] in [states[s[:2]] for s in geocodes]))):
                continue

            state_timeseries_coords_folder:str = "%s\\%s"%(timeseries_coords_folder, next(f for f in listdir(timeseries_coords_folder) if f.startswith(state)))

            states_irradiance[state[:2]] = defaultdict(dict[str, np.ndarray])

            cities_dicts:list[tuple[str, defaultdict]] = p.starmap(
                city_process,
                [(data_folder, ventures_folder, state_timeseries_coords_folder, state, city) for city in listdir('%s\\%s'%(ventures_folder, state)) if not(geocodes) or (city[1:8] in geocodes)]
            )

            coord_year_list:defaultdict[str, defaultdict[float, list[float]]]
            for city, coord_year_list in cities_dicts:
                states_irradiance[state[:2]][city[1:8]] = {coord:np.asarray([[year, power_qtd[0], power_qtd[1]] for year, power_qtd in year_list.items()]) for coord, year_list in coord_year_list.items()}

            print('%s processed'%(state[:2]))

    with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'wb', 32*1024*1024) as fout:
        pickle.dump(states_irradiance, fout)
    
    return states_irradiance