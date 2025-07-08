import numpy as np
from os import listdir
from os.path import dirname, abspath, isfile
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool
from os import makedirs
from collections import defaultdict
import pickle

from compiled_venture_functions import city_process, coord_generation

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

""" def city_process(main_folder:Path, ventures_folder:Path, state_timeseries_coords_folder:Path, state:str, city:str) -> tuple[str, defaultdict[str, defaultdict[str, list[np.int64]]]]:
    
    with open("%s\\%s\\%s"%(ventures_folder, state, city), 'br', 8*1024*1024) as file:
        text:bytes = file.read()
    ventures:list[str] = text.decode('utf-8').split('\r\n')[1:]

    city_timeseries_coords_file:Path = Path("%s\\%s"%(state_timeseries_coords_folder, next(f for f in listdir(state_timeseries_coords_folder) if f.startswith(city[:9]))))
    city_timeseries_coords:np.ndarray = np.loadtxt(city_timeseries_coords_file, np.float64, delimiter=',', ndmin=2, encoding='utf-8')[:, [1,0]]

    failty_coord:list[str] = []
    lat:list[str] = []
    lon:list[str] = []
    power:list[str] = []
    date:list[str] = []
    filtered_ventures:list[str] = []
    for line in ventures:
        try:
            venture_data:list[str] = line[1:-1].split('";"', 5)

            venture_data[2] = venture_data[2].replace(',', '.', 1)
            _ = float(venture_data[2])
            venture_data[3] = venture_data[3].replace(',', '.', 1)
            _ = float(venture_data[3])
            lat.append(venture_data[2])
            lon.append(venture_data[3])

            power.append(venture_data[4].replace(',', '', 1)+'0')
            date.append(venture_data[5].rsplit('";"', 1)[-1].replace('-', '', 2))
            filtered_ventures.append(line)

        except Exception:
            failty_coord.append(line)
    city_ventures_coords:np.ndarray = np.array([lat,lon], np.float64).T
    city_date_power:np.ndarray = np.array([date, power], np.int64).T
    sorted_args:np.ndarray = np.argsort(city_date_power[:, 0])
    city_ventures_coords = city_ventures_coords[sorted_args]
    city_date_power = city_date_power[sorted_args]

    if failty_coord:
        with open("%s\\outputs\\failty_coord.csv"%(main_folder), 'a', encoding='utf-8') as f:
            f.writelines(failty_coord)

    distances:np.ndarray
    idxs:np.ndarray
    distances, idxs = KDTree(city_timeseries_coords).query(city_ventures_coords, 1, workers=-1)

    faridxs:np.ndarray = np.argwhere(np.sqrt(np.sum((city_ventures_coords-city_timeseries_coords[idxs])**2, 1)) >= 0.03).T[0]
    if faridxs.shape[0]:
        makedirs("%s\\outputs\\Too Far Coords\\%s"%(main_folder, state), exist_ok=True)
        with open("%s\\outputs\\Too Far Coords\\%s\\%s-too-far.csv"%(main_folder, state, city[:9]), 'w', 1024*1024, encoding='utf-8') as f:
            f.write("source coord;closest timeseries coord;distance;line\n")
            f.writelines("(%7.2f,%6.2f) (%11.6f,%6.6f) %6.2f %s"%(*city_ventures_coords[i], *city_timeseries_coords[idxs[i]], distances[i], filtered_ventures[sorted_args[i]]) for i in faridxs)
    

    coord_date_list:defaultdict[str, defaultdict[str, list[np.int64]]] = defaultdict(defaultdict[str, list[np.int64]])
    for i in range(city_ventures_coords.shape[0]):
        venture_date:str = str(city_date_power[i, 0])
        timeseries_coord:str ='(%.6f,%.6f)'%(city_timeseries_coords[idxs[i], 0], city_timeseries_coords[idxs[i], 1])

        if (timeseries_coord not in coord_date_list):
            coord_date_list[timeseries_coord] = defaultdict(list[np.int64])

        if (venture_date not in coord_date_list[timeseries_coord]):
            coord_date_list[timeseries_coord][venture_date] = [np.int64(0), np.int64(0)]

        coord_date_list[timeseries_coord][venture_date][0] += city_date_power[i, 1]
        coord_date_list[timeseries_coord][venture_date][1] += 1
    
    return (city, coord_date_list) """

def ventures_process(sts:list[str] = [], geocodes:list[str] = []) -> defaultdict[str, defaultdict[str, dict[str, np.ndarray]]]:

    if (isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'))):
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'rb', 40*1024*1024) as fin:
            return pickle.load(fin)

    main_folder:Path = Path(dirname(abspath(__file__))).parent

    ventures_folder:Path = Path('%s\\data\\ventures'%(main_folder))

    timeseries_coords_folder:Path = Path('%s\\data\\timeseries_coords'%(main_folder))

    states_irradiance:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = defaultdict(defaultdict[str, dict[str, np.ndarray]])

    with Pool(cpu_count()+2) as p:

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

""" def coord_generation(geocode:str, coord:str, date_power_qty_array:np.ndarray, monolith:bool) -> dict[int, np.ndarray]:
    if (date_power_qty_array[0,0]>20241231):
        return {}
    
    if monolith:
        timeseries_path:Path =  Path('C:\\Programas\\timeseries\\%s\\monolith\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, coord))
    else:
        timeseries_path =  Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries\\%s\\[%s]\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, geocode, coord))
    
    with gzopen(timeseries_path, 'rb', 9) as f:
        text:bytes = f.read(16*1024*1024)
    lines:list[str] = text.decode('utf-8').splitlines()[9:-12]

    min_year:int = min(int(date_power_qty_array[0,0])//10000, 2020)
    i0:int = sum([366 if i%4==0 else 365 for i in range(int(lines[0][:4]), min_year)])*24

    col0:list[str] = []
    col1:list[float] = []
    col2:list[str] = []
    for line in lines[i0:]:
        spline = line.split(',', 6)
        col0.append(spline[0].replace(':', '', 1))
        col1.append(float(spline[1])+float(spline[2])+float(spline[3]))
        col2.append(spline[5])
    #timeseries_array:np.ndarray = np.array([col0, col1, col2], np.float64).T
    year_col:np.ndarray = np.array([col0], np.int64).T#timeseries_array[:, 0 ].astype(np.int64)
    val_cols:np.ndarray = np.array([col1, col2], np.float64).T#timeseries_array[:, 1:].astype(np.float64)


    year_filter:np.ndarray = year_col[:, 0]//(1_0000_0000)
    mask:list[np.ndarray] = [
        year_filter == 2020,
        year_filter == 2021,
        year_filter == 2022,
        year_filter == 2023
    ]

    c59_24:int = 59*24
    c60_24:int = 60*24

    arr2024_vals:np.ndarray  = np.zeros((366*24, 2), np.float64)
    arr2024_vals += val_cols[mask[0]]
    for i in [1, 2, 3]:
        year_vals:np.ndarray = val_cols[mask[i]]
        arr2024_vals[:c59_24] += year_vals[:c59_24]
        arr2024_vals[c60_24:] += year_vals[c59_24:]
        arr2024_vals[c59_24:c60_24] += np.mean([
            year_vals[c59_24-24:c59_24],
            year_vals[c59_24:c60_24]
        ], 0)
    arr2024_vals /= 4

    year_col = np.concatenate((year_col, 2024*(1_0000_0000) + year_col[mask[0]]%(1_0000_0000)), 0)
    val_cols = np.concatenate([val_cols, arr2024_vals], 0)

    
    if (states[geocode[:2]] == 'AC'):
        time_correction:int = 5
    elif (states[geocode[:2]] == 'AM'):
        time_correction = 4
    elif (geocode == '2605459'):
        time_correction = 2
    else:
        time_correction = 3
    
    min_date:np.int64 = np.argwhere(year_col//(1_0000) == date_power_qty_array[0, 0])[0, 0]
    dates:np.ndarray = year_col[min_date:].reshape((year_col.shape[0]-min_date)//24, 24, 1)
    shifted_vals:np.ndarray = np.concatenate([
        val_cols[min_date+time_correction:],
        val_cols[min_date:min_date+time_correction]
    ], 0).reshape((year_col.shape[0]-min_date)//24, 24, 2)
    
    irradiance:np.ndarray = shifted_vals[:, :, 0:1]
    t2m:np.ndarray = shifted_vals[:, :, 1:2]
    

    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))

    lossless_inst_power:np.ndarray = date_power_qty_array[:, 1].copy()
    date_power_qty_array[:, 1] = (np.round(date_power_qty_array[:, 1].astype(np.float64)*0.95*0.97*0.98, 0)).astype(np.int64)

    lossless_power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.int64)
    power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.float64)

    js:np.ndarray = np.argwhere(dates[:, 0, 0:1]//(1_0000) == date_power_qty_array[:, 0])[:, 0]
    for i in range(date_power_qty_array[date_power_qty_array[:, 0] < 20250101][:, 0].shape[0]):
        power_array[js[i]:, :] += (date_power_qty_array[i, 1]*np.power(0.995, np.arange(power_array[js[i]:].shape[0])/365.25))[:, np.newaxis, np.newaxis]
        lossless_power_array[js[i]:, :] += lossless_inst_power[i]

    energy_array:np.ndarray = np.concatenate([
        dates[:, :, :1],
        (irradiance[:, :, :1]*power_array*t_correction(irradiance[:, :, :1], t2m[:, :, :1]))/1000,
        irradiance[:, :, :1]*lossless_power_array/1000
    ], 2)
    
    energy_year:np.ndarray = energy_array[:, 0, 0]//(1_0000_0000)
    Z0:dict[int, np.ndarray] = {year:energy_array[energy_year == year] for year in range(int(energy_year[0]), int(energy_year[-1])+1)}
    
    return Z0 """

def plot_generation(state:dict[str, dict[str, np.ndarray]], period:tuple[int, int] = (0, 0), monolith:bool = False) -> None:
    state_abbreviation:str = states[list(state.keys())[0][:2]]

    preZ:dict[int, np.ndarray]
    if (not(isfile('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent)))):
        preZ = {}
        with Pool(cpu_count()+2) as p:
            all_coords_dict:dict[str, tuple[str, np.ndarray]] = {key:(geocode, coords_dict[key][coords_dict[key][:,0].argsort()]) for geocode, coords_dict in state.items() for key in coords_dict.keys()}
            
            Z0s:list[dict[int, np.ndarray]] = p.starmap(
                coord_generation,
                [(geocode__date_power_qty_array[0], coord, geocode__date_power_qty_array[1], monolith) for coord, geocode__date_power_qty_array in all_coords_dict.items()]
            )
            
        for Z0 in Z0s:
            for year, array in Z0.items():
                if year not in preZ:
                    preZ[year] = np.zeros((366 if year%4==0 else 365, 24, 3), np.float64)
                preZ[year][-array.shape[0]:, :, 1:] += array[:, :, 1:]
                preZ[year][-array.shape[0]:, :, 0]  = array[:, :, 0]

        with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'wb', 4*(1024**2)) as fout:
            pickle.dump(preZ, fout)
    else:
        with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'rb', 4*(1024**2)) as fin:
            preZ = pickle.load(fin)

    #all_coords_dict = {coord:coords_dict[coord][coords_dict[coord][:,0].argsort()] for coords_dict in state.values() for coord in coords_dict.keys()}
    #print(*['%4i, %5.2fMW, %5.2fMWh'%(key, sum([np.sum(array[array[:, 0]<=key][:, 1]) for array in all_coords_dict.values()])/(10**3), np.sum(preZ[key])/(10**6)) for key in sorted(preZ.keys())], sep='\n')

    #print([(key, np.round(np.sum(preZ[key])/(10**6),2)) for key in sorted(list(preZ.keys()))]) #(year, installed power [MW])

    Zm:np.ndarray = np.concatenate([preZ[key] for key in sorted(list(preZ.keys()))])

    if period == (0,0):
        period = (int(Zm[Zm[:,0,0]//(10**4) > 0][0, 0, 0]//1_0000), int(Zm[-1, 0, 0]//(1_0000)))
    
    i0:np.int64 = np.argwhere(Zm[:,0,0]//(10**4) >= period[0])[0, 0]
    i:np.int64 = np.argwhere(Zm[:,0,0]//(10**4) <= period[1])[-1, 0]
    Z:np.ndarray = Zm[i0:i+1]
    Z[:, :, 1:] = Z[:, :, 1:]/(10**6)
    usedZ:np.ndarray = Z[:, :, 1]#*1.125
    total_energy_produced:np.float64 = np.sum(usedZ)/10**6

    period = (int(Z[0, 0, 0]//1_0000), int(Z[-1, 0, 0]//1_0000))

    print('ventures:', period, Z[[0,-1], 0,  0].astype(np.str_))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #type: ignore
    from matplotlib.ticker import FixedLocator, FixedFormatter

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    Z_max_Y:np.ndarray = np.max(usedZ, axis=1)
    from scipy.ndimage import gaussian_filter1d
    smoothed_Z_max_Y = gaussian_filter1d(Z_max_Y, sigma=10)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, usedZ, cmap='viridis')
    ax.plot(x, [y.max()]*len(x), smoothed_Z_max_Y, color='#440154', linewidth=2, label='Smoothed Z Max over X')
    ax.view_init(20, -50, 0)

    all_idxs:list[int] = list(np.unique(Z[:, 0, 0]//1_00_0000, True)[1].astype(int))
    idxs:list[int] = [all_idxs[i] for i in range(0, len(all_idxs), len(all_idxs)//4)]
    month_labels = ['%04i-%02i'%(Z[i, 0, 0]//1_0000_0000, Z[i, 0, 0]%1_0000_0000/1_00_0000) for i in idxs]

    # Set the locations of the ticks to correspond to the first day of each unique month
    x_locator = FixedLocator(idxs)
    ax.xaxis.set_major_locator(x_locator)

    # Set the labels for these ticks using the month strings
    x_formatter = FixedFormatter(month_labels)
    ax.xaxis.set_major_formatter(x_formatter)

    ax.set_xlabel('Day of the Data [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Power [MW]')
    ax.set_title('Ventures PV Yield Across (%02i/%i:%02i/%i)\n[%s]\n\nTotal produced: %.2fTWh'%(period[0]%10000//100, period[0]//10000, period[1]%10000//100, period[1]//10000, state_abbreviation, total_energy_produced))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s\\outputs\\Ventures MMD PV Generation\\ventures-%s-(%i_%02i, %i_%02i).png"%(Path(dirname(abspath(__file__))).parent, state_abbreviation, period[0]//10000, period[0]%10000//100,  period[1]//10000, period[1]%10000//100), backend='Agg', dpi=200)
    plt.close()