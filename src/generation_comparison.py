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

from usine_generation import usines_plot, usines_pv_mmd_generation
from venture_generation import ventures_process

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

#get and plot ventures generation

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
        current_power = current_power*0.995 + (year_power_qty_array[year_power_qty_array[:, 0]==year][0, 1]*1000*0.95*0.97*0.98 if (year_power_qty_array[year_power_qty_array[:, 0]==year].size>0) else 0)
        
        year_mask:np.ndarray = timeseries_array[:, 0]//(10**8) == year
        
        Z0[year] = (timeseries_array[year_mask, 1]*current_power*t_correction(timeseries_array[year_mask, 1], timeseries_array[year_mask, 2])/1000).reshape((366 if year%4==0 else 365, 24))
    
    return Z0 

def plot_generation(state:dict[str, dict[str, np.ndarray]], monolith:bool = False) -> None:
    state_abbreviation:str = states[list(state.keys())[0][:2]]

    preZ:dict[int, np.ndarray]
    if (not(isfile('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent)))):
        preZ = {}
        with Pool(cpu_count()) as p:
            if monolith:
                all_coords_dict:dict[str, np.ndarray] = {key:coords_dict[key][coords_dict[key][:,0].argsort()] for coords_dict in state.values() for key in coords_dict.keys()}
                
                Z0s:list[dict[int, np.ndarray]] = p.starmap(coord_process_monolith, [(state_abbreviation, coord, year_power_qty_array) for coord, year_power_qty_array in all_coords_dict.items()])
                
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

    #all_coords_dict = {coord:coords_dict[coord][coords_dict[coord][:,0].argsort()] for coords_dict in state.values() for coord in coords_dict.keys()}
    #print(*['%4i, %5.2fMW, %5.2fMWh'%(key, sum([np.sum(array[array[:, 0]<=key][:, 1]) for array in all_coords_dict.values()])/(10**3), np.sum(preZ[key])/(10**6)) for key in sorted(preZ.keys())], sep='\n')
    
    
    if (state_abbreviation == 'AC'):
        time_correction:int = 5
    elif (state_abbreviation == 'AM'):
        time_correction = 4
    #elif (geocode == '2605459'):
    #    time_correction = 2
    else:
        time_correction = 3

    #print([(key, np.round(np.sum(preZ[key])/(10**6),2)) for key in sorted(list(preZ.keys()))]) #(year, installed power [MW])
    period:tuple[int, int] = (2016, 2018)
    Z:np.ndarray = np.concatenate([preZ[key] for key in range(period[0], period[1]+1)])/(10**6)
    Z = np.concatenate((Z[:, time_correction:], Z[:, :time_correction]), 1)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #type: ignore
    from matplotlib import use
    #use("Agg")

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Data [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Power [MW]')
    ax.set_title('Ventures PV Yield Across (%i:%i)\n[%s]\n\nTotal produced: %.2fTWh'%(*period, state_abbreviation, np.sum(Z)/10**6))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s\\outputs\\Ventures MMD PV Generation\\%s (%i, %i).png"%(Path(dirname(abspath(__file__))).parent, state_abbreviation, *period), backend='Agg', dpi=200)
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

    #Ventures build
    t0:float = perf_counter()
    states_cities_coords_array:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = ventures_process()
    print('Ventures process execution time:', perf_counter()-t0)

    #Save filtered coords
    """ t0 = perf_counter()
    save_timeseries_coords_filtered(states_cities_coords_array, True)
    print('Save filtered timeseries coords time:', perf_counter()-t0) """

    #Ventures coordinates data volume analysis
    """ total:int = 0
    for state, cities in states_cities_coords_array.items():
        subtotal:int = sum([len(city.keys()) for city in cities.values()])
        total += subtotal
        print('%s: %5i timeseries coords (%5.2f GiB compressed -> %5.1fmin)'%(state, subtotal, 1650*subtotal/(1024**2), 0.42981*subtotal/60))
    print('Total space required: %i timeseries (%.2f GiB after compression)'%(total, 1650*total/(1024**2)))
    print('Total minimum expected download time: Uncompressed %.2f TiB -> %.1fh'%(7.65*total/(1024**2), 0.42981*total/(60**2))) """


    #Ventures plot // depends on having timeseries
    t0 = perf_counter()
    plot_generation(states_cities_coords_array['SP'], True)
    print('Ventures generetaion plot execution time:', perf_counter()-t0)


    #Usines build
    t0 = perf_counter()
    per_state_usines_pv_mmd_generation:dict[str, np.ndarray] = usines_pv_mmd_generation()
    print('\nUsines process execution time:', perf_counter()-t0)

    #Usines plot
    t0 = perf_counter()
    usines_plot(list(per_state_usines_pv_mmd_generation.items()))
    print('Usines generation plot execution time:', perf_counter()-t0)

if __name__ == "__main__":
    main()
    #usine_generation('SP')

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')