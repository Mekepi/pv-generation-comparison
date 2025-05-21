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
from mpl_toolkits.mplot3d import Axes3D #type: ignore
import pickle

from usine_generation import usine_plot, usines_pv_mmd_generation

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

#ventures

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

    return states_irradiance

def coord_process(geocode:str, coord:str, year_power_qty_array:np.ndarray) -> dict[int, np.ndarray]:
    timeseries_path:Path =  Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries\\%s\\[%s]\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, geocode, coord))
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]

    lines_array:np.ndarray = np.asarray([
        [
            int(''.join(line.split(',')[0].split(':'))),
            sum([float(v) for v in line.split(',')[1:4]]),
            float(line.split(',')[5])
        ]
        for line in lines if int(line[:4]) >= int(year_power_qty_array[0,0])])

    current_year:int = int(year_power_qty_array[0,0])
    current_power:float = year_power_qty_array[0,1]*0.95*0.97*0.98
    j:int = 0

    def t_correction(g:float, t2m:float) -> float:
        Tc:float = (t2m+g*25/800)
        Tc = Tc if Tc>25 else 25
        
        return (1 - 0.0045*(Tc-25))
    
    Z0:dict[int, np.ndarray] = {}
    for i in range(0, lines_array.shape[0], 24):
        if (lines_array[i, 0]//(10**8) > current_year):
            current_year = lines_array[i, 0]//(10**8)
            current_power = current_power*0.995 + year_power_qty_array[year_power_qty_array[:, 0]==current_year][0, 1]*0.95*0.97*0.98 if (year_power_qty_array[year_power_qty_array[:, 0]==current_year].size>0) else 0
            j = 0
        
        if (current_year not in Z0):
            Z0[current_year] = np.zeros((366 if current_year%4==0 else 365, 24))

        day_irradiation_temperature:np.ndarray = lines_array[i:i+24, 1:]

        try: Z0[current_year][j] += (day_irradiation_temperature[:, 0]*current_power/1000)*np.vectorize(t_correction)(day_irradiation_temperature[:,0], day_irradiation_temperature[:,1])
        except Exception as e:
            print(j, current_year, 'bissexto' if current_year%4==0 else 'normal', Z0[current_year].shape, lines_array[i])
            raise e
        j += 1
    
    return Z0 

def coord_process_monolith(state:str, coord:str, year_power_qty_array:np.ndarray) -> dict[int, np.ndarray]:
    if (int(year_power_qty_array[0,0])>2024):
        return {}
    
    timeseries_path:Path =  Path('C:\\Programas\\timeseries\\%s\\monolith\\timeseries%s.csv.gz'%(state, coord))
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]
    
    min_year:int = min([int(year_power_qty_array[0,0]), 2020])
    timeseries_array:np.ndarray = np.asarray([
        [
            line.split(',')[0].replace(':', ''), #YYYYMMDDhhmm
            sum([float(v) for v in line.split(',')[1:4]]), #summed G's
            line.split(',')[5] #T2m
        ]
        for line in lines if int(line[:4]) >= min_year], np.float64)

    arr2024:np.ndarray = np.zeros((366*24, 3))
    arr2024[:, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == 2020][:, 1:]
    for i in [2021, 2022, 2023]:
        arr2024[:59*24, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == i][:59*24, 1:]
        arr2024[60*24:, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == i][59*24:, 1:]
        arr2024[59*24:60*24, 1:] += (timeseries_array[timeseries_array[:, 0]//(10**8) == i][58*24:59*24, 1:]+timeseries_array[timeseries_array[:, 0]//(10**8) == i][59*24:60*24, 1:])/2
    
    arr2024[:, 0] = timeseries_array[timeseries_array[:, 0]//(10**8) == 2020][:, 0]%(10**8)
    arr2024[:, 0] += 2024*(10**8)

    arr2024[:, 1:] /= 4

    timeseries_array = np.concatenate((timeseries_array, arr2024))

    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))
    
    Z0:dict[int, np.ndarray] = {}
    current_power:float = 0
    for year in range(min_year, int(timeseries_array[-1, 0]//(10**8))+1):
        current_power = current_power*0.995 + (year_power_qty_array[year_power_qty_array[:, 0]==year][0, 1]*1000*0.95*0.97*0.98 if (year_power_qty_array[year_power_qty_array[:, 0]==year].size>0) else 0)
        
        year_mask:np.ndarray = timeseries_array[:, 0]//(10**8) == year
        
        Z0[year] = (timeseries_array[year_mask, 1]*current_power*t_correction(timeseries_array[year_mask, 1], timeseries_array[year_mask, 2])/1000).reshape((366 if year%4==0 else 365, 24))
    
    return Z0 

def plot_generation(state:dict[str, dict[str, np.ndarray]], monolith:bool = False) -> None:
    """ all_coords_dict:dict[str, np.ndarray] = {}
    [all_coords_dict.update([(key, coords_dict[key][coords_dict[key][:,0].argsort()]) for key in coords_dict.keys()]) for coords_dict in state.values()]
    print(type(list(all_coords_dict.items())[0][1][0,0]), list(all_coords_dict.items())[0][1][0,0])
    print(*[(coord, array[0,0], array[-1,0], array[:, 1])for coord, array in sorted(list(all_coords_dict.items()), key=lambda e: e[1][0,0])[:30]], sep='\n')
    preZ = coord_process_monolith(states[list(state.keys())[0][:2]], '(-21.210883,-47.794643)', all_coords_dict['(-21.210883,-47.794643)'])
    #print(coord_process_monolith(states[list(state.keys())[0][:2]], '(-21.210883,-47.794643)', all_coords_dict['(-21.210883,-47.794643)'])) """

    preZ:dict[int, np.ndarray]
    if (not(isfile('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent)))):
        preZ = {}
        with Pool(cpu_count()) as p:
            if monolith:
                all_coords_dict:dict[str, np.ndarray] = {key:coords_dict[key][coords_dict[key][:,0].argsort()] for coords_dict in state.values() for key in coords_dict.keys()}
                
                Z0s:list[dict[int, np.ndarray]] = p.starmap(coord_process_monolith, [(states[list(state.keys())[0][:2]], coord, year_power_qty_array) for coord, year_power_qty_array in all_coords_dict.items()])
                
                for Z0 in Z0s:
                    for year, array in Z0.items():
                        if year not in preZ:
                            preZ[year] = np.zeros_like(array)
                        preZ[year] += array
            else:
                for geocode, coords_dict in state.items():
                    Z0s = p.starmap(coord_process, [(geocode, coord, year_power_qty_array) for coord, year_power_qty_array in coords_dict.items()])
                    
                    for Z0 in Z0s:
                        for year, array in Z0.items():
                            if year not in preZ:
                                preZ[year] = np.zeros_like(array)
                            preZ[year] += array

            with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'wb', 50*(1024**2)) as fout:
                pickle.dump(preZ, fout)
    else:
        with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'rb', 50*(1024**2)) as fin:
            preZ = pickle.load(fin)

    all_coords_dict = {coord:coords_dict[coord][coords_dict[coord][:,0].argsort()] for coords_dict in state.values() for coord in coords_dict.keys()}
    print(*['%4i, %5.2fMW, %5.2fMWh'%(key, sum([np.sum(array[array[:, 0]<=key][:, 1]) for array in all_coords_dict.values()])/(10**3), np.sum(preZ[key])/(10**6)) for key in sorted(preZ.keys())], sep='\n')
    
    state_abbreviation:str = states[list(state.keys())[0][:2]]
    if (state_abbreviation == 'AC'):
        time_correction:int = 5
    elif (state_abbreviation == 'AM'):
        time_correction = 4
    #elif (geocode == '2605459'):
    #    time_correction = 2
    else:
        time_correction = 3

    #print([(key, np.round(np.sum(preZ[key])/1000,2)) for key in sorted(list(preZ.keys()))])
    Z:np.ndarray = np.concatenate([preZ[key] for key in [2023,2024]])/(10**6)
    Z = np.concatenate((Z[:, time_correction:], Z[:, :time_correction]), 1)

    import matplotlib.pyplot as plt
    from matplotlib import use
    #use("Agg")

    x:np.ndarray = np.array(list(range(1, Z.shape[0]+1)))
    y:np.ndarray = np.array(list(range(Z.shape[1])))

    Y, X = np.meshgrid(y, x)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Data [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Power [MW]')
    ax.set_title('Ventures PV Yield Across (%i:%i)\n[%s]\n\nTotal produced: %.2fTWh'%(2023, 2024, states[list(state.keys())[0][:2]], np.sum(Z)/10**6))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s\\outputs\\Ventures MMD PV Generation\\%s.png"%(Path(dirname(abspath(__file__))).parent, states[list(state.keys())[0][:2]]), backend='Agg', dpi=200)
    plt.close()

def save_timeseries_coords_filtered(states_cities_coords_array:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]], monolith:bool=False) -> None:
    for state, cities in states_cities_coords_array.items():
        makedirs('%s\\data\\timeseries_coords_filtered\\%s'%(Path(dirname(abspath(__file__))).parent, state), exist_ok=True)
        
        if monolith:
            with open('%s\\data\\timeseries_coords_filtered\\%s\\%s_coords.csv'%(Path(dirname(abspath(__file__))).parent, state, state), 'w', 1024**2, 'utf-8') as fout:
                for coords in cities.values():
                    fout.writelines([coord[1:-1]+'\n' for coord in coords.keys()])
        else:
            for geocode, coords in cities.items():
                with open('%s\\data\\timeseries_coords_filtered\\%s\\[%s]_coords.csv'%(Path(dirname(abspath(__file__))).parent, state, geocode), 'w', 1024**2, 'utf-8') as fout:
                    fout.writelines([coord[1:-1]+'\n' for coord in coords.keys()])

def main(sts:list[str] = [], geocodes:list[str] = []) -> None:

    #Ventures

    t0:float = perf_counter()
    if (not(isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl')))):
        states_cities_coords_array:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = ventures_process()
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'wb', 20*1024*1024) as fout:
            pickle.dump(states_cities_coords_array, fout)
    else:
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\states_cities_coords_array.pkl'), 'rb', 20*1024*1024) as fin:
            states_cities_coords_array = pickle.load(fin)
    print('Ventures process execution time:', perf_counter()-t0)

    """ t0 = perf_counter()
    save_timeseries_coords_filtered(states_cities_coords_array, True)
    print('Save filtered timeseries coords time:', perf_counter()-t0) """

    #print(states_cities_coords_array.keys())
    #print(states_cities_coords_array['SP'].keys())
    #print(sorted([(coord, array[0,0], array[0, 1]) for coord, array in  states_cities_coords_array['SP']['3550308'].items()], key=lambda e: e[1]))
    #print(list(states_cities_coords_array['SP']['3550308'].keys())[0])

    #Ventures coordinates data volume analysis

    """ total:int = 0
    for state, cities in states_cities_coords_array.items():
        subtotal:int = sum([len(city.keys()) for city in cities.values()])
        total += subtotal
        print('%s: %5i timeseries coords (%5.2f GiB compressed -> %5.1fmin)'%(state, subtotal, 1650*subtotal/(1024**2), 0.42981*subtotal/60))
    print('Total space required: %i timeseries (%.2f GiB after compression)'%(total, 1650*total/(1024**2)))
    print('Total minimum expected download time: Uncompressed %.2f TiB -> %.1fh'%(7.65*total/(1024**2), 0.42981*total/(60**2))) """

    #Ventures plot

    t0 = perf_counter()
    plot_generation(states_cities_coords_array['SP'], True)
    print('Ventures generetaion plot execution time:', perf_counter()-t0)

    #Usines

    t0 = perf_counter()
    if (not(isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl')))):
        per_state_usines_pv_mmd_generation:dict[str, np.ndarray] = usines_pv_mmd_generation()
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'), 'wb', 6*1024*1024) as fout:
            pickle.dump(per_state_usines_pv_mmd_generation, fout)
    else:
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'), 'rb', 6*1024*1024) as fin:
            per_state_usines_pv_mmd_generation = pickle.load(fin)
    print('Usines process execution time:', perf_counter()-t0)

    #print(*["(%s, %s)\n"%(state, str(array.shape)) for state, array in per_state_usines_pv_mmd_generation.items()])
    usine_plot('SP', per_state_usines_pv_mmd_generation['SP'])

    #continuar

if __name__ == "__main__":
    main()
    #usine_generation('SP')

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')