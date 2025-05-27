import numpy as np
from time import perf_counter
from os import listdir
from os.path import dirname, abspath, isfile
from pathlib import Path
from scipy.spatial import KDTree
from gzip import open as gzopen
from os import makedirs
from collections import defaultdict

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def city_process(main_folder:Path, ventures_folder:Path, state_timeseries_coords_folder:Path, state:str, city:str) -> tuple[str, defaultdict[str, defaultdict[str, list[np.int64]]]]:
    
    with open("%s\\%s\\%s"%(ventures_folder, state, city), 'r', 8*1024*1024, encoding='utf-8') as file:
        ventures:list[str] = file.readlines()[1:]

    city_timeseries_coords_file:Path = Path("%s\\%s"%(state_timeseries_coords_folder, next(f for f in listdir(state_timeseries_coords_folder) if f.startswith(city[:9]))))
    city_timeseries_coords:np.ndarray = np.loadtxt(city_timeseries_coords_file, np.float64, delimiter=',', ndmin=2, encoding='utf-8')[:, [1,0]]

    failty_coord:list[str] = []
    lat:list[float] = []
    lon:list[float] = []
    power:list[str] = []
    date:list[str] = []
    filtered_ventures:list[str] = []
    for line in ventures:
        try:
            venture_data:list[str] = line[1:-2].split('";"', 5)

            venture_data[2] = venture_data[2].replace(',', '.', 1)
            _ = float(venture_data[2])
            venture_data[3] = venture_data[3].replace(',', '.', 1)
            _ = float(venture_data[3])
            lat.append(float(venture_data[2]))
            lon.append(float(venture_data[3]))

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
    
    return (city, coord_date_list)

def coord_process(geocode:str, coord:str, date_power_qty_array:np.ndarray) -> dict[int, np.ndarray]:
    timeseries_path:Path =  Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries\\%s\\[%s]\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, geocode, coord))
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]

    min_year:int = min([int(date_power_qty_array[0,0]), 2020])
    timeseries_array:np.ndarray = np.zeros((sum([366 if i%4==0 else 365 for i in range(min_year, 2024)])*24, 3), np.float64)
    i0:int = sum([366 if i%4==0 else 365 for i in range(2005, min_year)])*24
    j:int = 0
    for line in lines[i0:]:
        spline:list[str] = line.split(',', 6)[:-1]
        timeseries_array[j, 0] = spline[0].replace(':', '')
        timeseries_array[j, 1] = sum([float(v) for v in spline[1:4]])
        timeseries_array[j, 2] = spline[5]
        j += 1

    
    arr2024:np.ndarray = np.zeros((366*24, 3))

    arr2024[:, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == 2020][:, 1:]
    for i in [2021, 2022, 2023]:
        arr2024[:59*24, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == i][:59*24, 1:]
        arr2024[60*24:, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == i][59*24:, 1:]
        arr2024[59*24:60*24, 1:] += (timeseries_array[timeseries_array[:, 0]//(10**8) == i][58*24:59*24, 1:]+timeseries_array[timeseries_array[:, 0]//(10**8) == i][59*24:60*24, 1:])/2
    arr2024[:, 1:] /= 4

    arr2024[:, 0] = timeseries_array[timeseries_array[:, 0]//(10**8) == 2020][:, 0]%(10**8)
    arr2024[:, 0] += 2024*(10**8)
    
    timeseries_array = np.concatenate((timeseries_array, arr2024))


    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))
    
    Z0:dict[int, np.ndarray] = {}
    current_power:float = 0.
    for year in range(min_year, int(timeseries_array[-1, 0]//(10**8))+1):
        current_power = current_power*0.995 + (date_power_qty_array[date_power_qty_array[:, 0]==year][0, 1]*1000*0.95*0.97*0.98 if (date_power_qty_array[date_power_qty_array[:, 0]==year].size>0) else 0)
        
        year_mask:np.ndarray = timeseries_array[:, 0]//(10**8) == year
        
        Z0[year] = (timeseries_array[year_mask, 1]*current_power*t_correction(timeseries_array[year_mask, 1], timeseries_array[year_mask, 2])/1000).reshape((366 if year%4==0 else 365, 24))
    
    return Z0 

def coord_process_monolith(geocode:str, coord:str, date_power_qty_array:np.ndarray) -> dict[int, np.ndarray]:
    if (date_power_qty_array[0,0]>20241231):
        return {}
    
    timeseries_path:Path =  Path('C:\\Programas\\timeseries\\%s\\monolith\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, coord))
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]

    min_year:int = int(min(date_power_qty_array[0,0]//10000, 2020))
    i0:int = sum([366 if i%4==0 else 365 for i in range(int(lines[0][:4]), min_year)])*24

    col0:list[str] = []
    col1:list[float] = []
    col2:list[str] = []
    for line in lines[i0:]:
        spline = line.split(',', 6)
        col0.append(spline[0].replace(':', '', 1))
        col1.append(float(spline[1])+float(spline[2])+float(spline[3]))
        col2.append(spline[5])
    timeseries_array:np.ndarray = np.array([col0, col1, col2], np.float64).T

    
    year_col:np.ndarray = timeseries_array[:, 0 ].astype(np.int64)
    val_cols:np.ndarray = timeseries_array[:, 1:].astype(np.float64)

    year_filter:np.ndarray = year_col//(1_0000_0000)
    mask:list[np.ndarray] = [
        year_filter == 2020,
        year_filter == 2021,
        year_filter == 2022,
        year_filter == 2023
    ]

    c59_24:int = 59*24
    c60_24:int = 60*24

    arr2024:np.ndarray = np.zeros((366*24, 3), np.float64)

    arr2024[:, 1:] += val_cols[mask[0]]
    for i in [1, 2, 3]:
        year_vals:np.ndarray = val_cols[mask[i]]
        arr2024[:c59_24, 1:] += year_vals[:c59_24]
        arr2024[c60_24:, 1:] += year_vals[c59_24:]
        arr2024[c59_24:c60_24, 1:] += np.mean([
            year_vals[c59_24-24:c59_24],
            year_vals[c59_24:c60_24]
        ], 0)
    
    arr2024 /= 4

    arr2024[:, 0] = 2024*(1_0000_0000) + year_col[mask[0]]%(1_0000_0000)

    timeseries_array = np.concatenate([timeseries_array, arr2024], 0)

    
    if (states[geocode[:2]] == 'AC'):
        time_correction:int = 5
    elif (states[geocode[:2]] == 'AM'):
        time_correction = 4
    elif (geocode == '2605459'):
        time_correction = 2
    else:
        time_correction = 3
    
    timeseries_mask_arg:np.int64 = np.argwhere(timeseries_array[:, 0]//(10**4) == date_power_qty_array[0, 0])[0, 0]
    
    timeseries_array[timeseries_mask_arg:, 1:] = np.concatenate([
        timeseries_array[timeseries_mask_arg+time_correction:, 1:],
        timeseries_array[timeseries_mask_arg:timeseries_mask_arg+time_correction, 1:]
    ], 0)

    date_irradiance_t2m_array = timeseries_array[timeseries_mask_arg:].reshape(timeseries_array[timeseries_mask_arg:].shape[0]//24, 24, 3)


    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))

    #lossless_inst_power:np.ndarray = date_power_qty_array[:, 1].copy()
    date_power_qty_array[:, 1] = (np.round(date_power_qty_array[:, 1].astype(np.float64)*0.95*0.97*0.98, 0)).astype(np.int64)

    #lossless_power_array:np.ndarray = np.zeros((date_irradiance_t2m_array.shape[0], 24, 1), np.int64)
    power_array:np.ndarray = np.zeros((date_irradiance_t2m_array.shape[0], 24, 1), np.float64)
    js:np.ndarray = np.argwhere((date_irradiance_t2m_array[:, 0, 0]//(1_0000))[:, np.newaxis] == date_power_qty_array[:, 0])[:, 0]
    for i in range(date_power_qty_array[date_power_qty_array[:, 0] < 20250101][:, 0].shape[0]):
        power_array[js[i]:, :] += (date_power_qty_array[i, 1]*np.power(0.995, np.arange(power_array[js[i]:].shape[0])/365.25))[:, np.newaxis, np.newaxis]
        #lossless_power_array[js[i]:, :] += lossless_inst_power[i]

    energy_array:np.ndarray = np.concatenate([
        date_irradiance_t2m_array[:, :, :1],
        (date_irradiance_t2m_array[:, :, 1:2]*power_array*t_correction(date_irradiance_t2m_array[:, :, 1:2], date_irradiance_t2m_array[:, :, 2:]))/1000
    ], 2)

    """ lossless_energy:np.ndarray = date_irradiance_t2m_array[:, :, 1:2]*lossless_power_array/1000
    loss_array:np.ndarray = (energy_array[energy_array[:, :, 1]>1e-16][:, 1]/lossless_energy[lossless_energy>1e-16])
    print(loss_array, np.mean(loss_array)) """
    
    energy_year:np.ndarray = energy_array[:, 0, 0]//(10**8)
    Z0:dict[int, np.ndarray] = {year:energy_array[energy_year == year] for year in range(int(energy_year[0]), int(energy_year[-1])+1)}
    
    return Z0 