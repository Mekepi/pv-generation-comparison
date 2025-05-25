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

def city_process(main_folder:Path, ventures_folder:Path, state_timeseries_coords_folder:Path, state:str, city:str) -> tuple[str, defaultdict[str, defaultdict[str, list[int]]]]:

    with open("%s\\%s\\%s"%(ventures_folder, state, city), 'r', 8*1024*1024, encoding='utf-8') as file:
        ventures:list[str] = file.readlines()[1:]

    city_timeseries_coords_file:Path = Path("%s\\%s"%(state_timeseries_coords_folder, next(f for f in listdir(state_timeseries_coords_folder) if f.startswith(city[:9]))))
    city_timeseries_coords:np.ndarray = np.loadtxt(city_timeseries_coords_file, delimiter=',', ndmin=2, encoding='utf-8')

    failty_coord:list[str] = []
    city_ventures_coords_list:list[list[float]] = []
    for i in range(len(ventures)):
        venture_data:list[str] = ventures[i].split('";"')
        try: city_ventures_coords_list.append([float(venture_data[3].replace(',', '.')), float(venture_data[2].replace(',', '.')), float(i)])
        except Exception:
            failty_coord.append(ventures[i])
    city_ventures_coords:np.ndarray = np.asarray(city_ventures_coords_list)

    if failty_coord:
        with open("%s\\outputs\\failty_coord.csv"%(main_folder), 'a', encoding='utf-8') as f:
            f.writelines(failty_coord)

    distances:list[float]
    idxs:list[int]
    distances, idxs = cKDTree(city_timeseries_coords).query(city_ventures_coords[:, :2], 1, workers=-1) # type: ignore

    faridxs:list[str] = ["(%7.2f,%6.2f) (%11.6f,%6.6f) %6.2f    %s"%(*city_ventures_coords[:, :2][i],*city_timeseries_coords[idxs[i]],distances[i],ventures[i]) for i in range(len(distances)) if distances[i]>=0.03]
    
    if faridxs:
        makedirs("%s\\outputs\\Too Far Coords\\%s"%(main_folder, state), exist_ok=True)
        with open("%s\\outputs\\Too Far Coords\\%s\\%s-too-far.csv"%(main_folder, state, city[:9]), 'w', 1024*1024*256, encoding='utf-8') as f:
            f.write("source coord;closest timeseries coord;distance;line\n")
            f.writelines(faridxs)
    
    coord_date_list:defaultdict[str, defaultdict[str, list[int]]] = defaultdict(defaultdict[str, list[int]])

    filtered_ventures:np.ndarray = np.asarray([v[1:-2].split('";"') for v in ventures])[city_ventures_coords[:,2].astype(int)]
    filtered_ventures = np.concatenate([filtered_ventures, city_timeseries_coords[idxs]], 1)
    filtered_ventures = filtered_ventures[filtered_ventures[:, -3].argsort()]
    filtered_ventures[:, -3] = np.char.replace(filtered_ventures[:, -3], '-', '')
    filtered_ventures[:,  4] = np.char.replace(filtered_ventures[:,  4], ',', '.')
    
    for i in range(filtered_ventures.shape[0]):
        venture_data = filtered_ventures[i]
        timeseries_coord:str ='(%.6f,%.6f)'%(float(filtered_ventures[i, -1]), float(filtered_ventures[i, -2]))

        if (timeseries_coord not in coord_date_list):
            coord_date_list[timeseries_coord] = defaultdict(list[int])

        if (venture_data[27] not in coord_date_list[timeseries_coord]):
            coord_date_list[timeseries_coord][venture_data[27]] = [0, 0]

        coord_date_list[timeseries_coord][venture_data[27]][0] += int(float(venture_data[4])*1000)
        coord_date_list[timeseries_coord][venture_data[27]][1] += 1
    
    return (city, coord_date_list)

def ventures_process(sts:list[str] = [], geocodes:list[str] = []) -> defaultdict[str, defaultdict[str, dict[str, np.ndarray]]]:

    if (isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'))):
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'rb', 70*1024*1024) as fin:
            return pickle.load(fin)

    main_folder:Path = Path(dirname(abspath(__file__))).parent

    ventures_folder:Path = Path('%s\\data\\ventures'%(main_folder))

    timeseries_coords_folder:Path = Path('%s\\data\\timeseries_coords'%(main_folder))

    states_irradiance:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = defaultdict(defaultdict[str, dict[str, np.ndarray]])

    with Pool(cpu_count()) as p:

        with open("%s\\outputs\\failty_coord.csv"%(main_folder), 'w', encoding='utf-8') as f:
            f.close()
        
        for state in states.values():

            if ((sts and not(state[:2] in sts)) or (geocodes and not(state[:2] in [states[s[:2]] for s in geocodes]))):
                continue

            state_timeseries_coords_folder:str = "%s\\%s"%(timeseries_coords_folder, next(f for f in listdir(timeseries_coords_folder) if f.startswith(state)))

            states_irradiance[state[:2]] = defaultdict(dict[str, np.ndarray])

            cities_dicts:list[tuple[str, defaultdict[str, defaultdict[str, list[int]]]]] = p.starmap(
                city_process,
                [(main_folder, ventures_folder, state_timeseries_coords_folder, state, city) for city in listdir('%s\\%s'%(ventures_folder, state)) if not(geocodes) or (city[1:8] in geocodes)]
            )
            
            for city, coord_date_list in cities_dicts:
                states_irradiance[state[:2]][city[1:8]] = {coord:np.asarray([[date, power_qtd[0], power_qtd[1]] for date, power_qtd in year_list.items()], np.int64) for coord, year_list in coord_date_list.items()}

            print('%s processed'%(state[:2]))

    with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'wb', 32*1024*1024) as fout:
        pickle.dump(states_irradiance, fout)
    
    return states_irradiance

def save_ventures_timeseries_coords_filtered(states_cities_coords_array:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]], monolith:bool=False) -> None:
    '''
    If monolith is activeded, it creates a monolith coordinates file instead of separating them into per city files.
    '''
    for state, cities in states_cities_coords_array.items():
        makedirs('%s\\data\\timeseries_coords_filtered\\%s'%(Path(dirname(abspath(__file__))).parent, state), exist_ok=True)
        
        if monolith:
            with open('%s\\data\\timeseries_coords_filtered\\%s\\%s_coords.csv'%(Path(dirname(abspath(__file__))).parent, state, state), 'w', 1024**2, 'utf-8') as fout:
                for geocode, coords in cities.items():
                    fout.writelines(['%s,%s\n'%(coord[1:-1], geocode) for coord in coords.keys()])
        else:
            for geocode, coords in cities.items():
                with open('%s\\data\\timeseries_coords_filtered\\%s\\[%s]_coords.csv'%(Path(dirname(abspath(__file__))).parent, state, geocode), 'w', 1024**2, 'utf-8') as fout:
                    fout.writelines([coord[1:-1]+'\n' for coord in coords.keys()])