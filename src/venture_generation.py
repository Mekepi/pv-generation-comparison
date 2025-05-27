import numpy as np
from time import perf_counter
from os import listdir
from os.path import dirname, abspath, isfile
from pathlib import Path
from scipy.spatial import KDTree
from psutil import cpu_count
from multiprocessing import Pool
from gzip import open as gzopen
from os import makedirs
from collections import defaultdict
import pickle

from functions.venture_functions import city_process, coord_process_monolith, coord_process

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def ventures_process(sts:list[str] = [], geocodes:list[str] = []) -> defaultdict[str, defaultdict[str, dict[str, np.ndarray]]]:

    if (isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'))):
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'rb', 40*1024*1024) as fin:
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

            cities_dicts:list[tuple[str, defaultdict[str, defaultdict[str, list[np.int64]]]]] = p.starmap(
                city_process,
                [(main_folder, ventures_folder, state_timeseries_coords_folder, state, city) for city in listdir('%s\\%s'%(ventures_folder, state)) if not(geocodes) or (city[1:8] in geocodes)]
            )
            
            for city, coord_date_list in cities_dicts:
                states_irradiance[state[:2]][city[1:8]] = {
                    coord:np.array([[date, power_qtd[0], power_qtd[1]] for date, power_qtd in year_list.items()], np.int64
                    ) for coord, year_list in coord_date_list.items()
                }

            print('%s processed'%(state[:2]))

    with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'wb', 40*1024*1024) as fout:
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

def plot_generation(state:dict[str, dict[str, np.ndarray]], period:tuple[int, int] = (0, 0), monolith:bool = False) -> None:
    state_abbreviation:str = states[list(state.keys())[0][:2]]

    preZ:dict[int, np.ndarray]
    if (not(isfile('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent)))):
        preZ = {}
        with Pool(cpu_count()) as p:
            if monolith:
                all_coords_dict:dict[str, tuple[str, np.ndarray]] = {key:(geocode, coords_dict[key][coords_dict[key][:,0].argsort()]) for geocode, coords_dict in state.items() for key in coords_dict.keys()}
                
                Z0s:list[dict[int, np.ndarray]] = p.starmap(
                    coord_process_monolith,
                    [(geocode__date_power_qty_array[0], coord, geocode__date_power_qty_array[1]) for coord, geocode__date_power_qty_array in all_coords_dict.items()]
                )
                
                for Z0 in Z0s:
                    for year, array in Z0.items():
                        if year not in preZ:
                            preZ[year] = np.zeros((366 if year%4==0 else 365, 24, 2), np.float64)
                        preZ[year][-array.shape[0]:, :, 1] += array[:, :, 1]
                        preZ[year][-array.shape[0]:, :, 0]  = array[:, :, 0]

            else:
                    for geocode, coords_dict in state.items():
                        Z0s = p.starmap(coord_process, [(geocode, coord, date_power_qty_array) for coord, date_power_qty_array in coords_dict.items()])
                        
                        for Z0 in Z0s:
                            for year, array in Z0.items():
                                if year not in preZ:
                                    preZ[year] = np.zeros((366 if year%4==0 else 365, 24, 2))
                                preZ[year][-array.shape[0]:] += array

            with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'wb', 4*(1024**2)) as fout:
                pickle.dump(preZ, fout)
    else:
        with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'rb', 4*(1024**2)) as fin:
            preZ = pickle.load(fin)

    #all_coords_dict = {coord:coords_dict[coord][coords_dict[coord][:,0].argsort()] for coords_dict in state.values() for coord in coords_dict.keys()}
    #print(*['%4i, %5.2fMW, %5.2fMWh'%(key, sum([np.sum(array[array[:, 0]<=key][:, 1]) for array in all_coords_dict.values()])/(10**3), np.sum(preZ[key])/(10**6)) for key in sorted(preZ.keys())], sep='\n')

    #print([(key, np.round(np.sum(preZ[key])/(10**6),2)) for key in sorted(list(preZ.keys()))]) #(year, installed power [MW])

    if period == (0,0):
        period_list:list[int] = sorted(list(preZ.keys()))
        period = (period_list[0]*(10**4)+101, period_list[-1]*(10**4)+1231)

    Zm:np.ndarray = np.concatenate([preZ[key] for key in sorted(list(preZ.keys()))])

    i0:np.int64 = np.argwhere(Zm[:,0,0]//(10**4) >= period[0])[0, 0]
    i:np.int64 = np.argwhere(Zm[:,0,0]//(10**4) <= period[1])[-1, 0]
    Z:np.ndarray = Zm[i0:i+1]
    Z[:, :, 1] = Z[:, :, 1]/(10**6)*1.125

    print('ventures:', period, Z[[0,-1], 0,  0].astype(np.str_))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #type: ignore

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    Z_max_Y:np.ndarray = np.max(Z[:, :, 1], axis=1)
    from scipy.ndimage import gaussian_filter1d
    smoothed_Z_max_Y = gaussian_filter1d(Z_max_Y, sigma=10)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z[:, :, 1], cmap='viridis')
    ax.plot(x, [y.max()]*len(x), smoothed_Z_max_Y, color='#440154', linewidth=2, label='Smoothed Z Max over X')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Data [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Power [MW]')
    ax.set_title('Ventures PV Yield Across (%02i/%i:%02i/%i)\n[%s]\n\nTotal produced: %.2fTWh'%(period[0]%10000//100, period[0]//10000, period[1]%10000//100, period[1]//10000, state_abbreviation, np.sum(Z[:, :, 1])/10**6))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s\\outputs\\Ventures MMD PV Generation\\ventures-%s-(%i_%02i, %i_%02i).png"%(Path(dirname(abspath(__file__))).parent, state_abbreviation, period[0]//10000, period[0]%10000//100,  period[1]//10000, period[1]%10000//100), backend='Agg', dpi=200)
    plt.close()