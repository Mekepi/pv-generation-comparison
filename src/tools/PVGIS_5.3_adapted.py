from urllib3 import request
from multiprocessing import Process, Pipe, active_children
from multiprocessing.connection import PipeConnection
from time import perf_counter, sleep
from os import remove, makedirs, listdir, rename
from os.path import dirname, abspath, isfile, getsize
from pathlib import Path
from psutil import virtual_memory
from gzip import open as gsopen

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def compress_file(file_path:Path):
    with open(file_path, "rb") as fin:
        with gsopen("%s.gz"%(str(file_path)), "wb", 9,) as fout:
            fout.write(fin.read())
    remove(file_path)

def request_timeseries(lat:str, lon:str, geocode:str, con:PipeConnection, compressed:bool = True, monolith:bool = False) -> None:
    line:str = ','.join((lat, lon, geocode))
    inner_folder:str = 'monolith' if monolith else '[%s]'%(geocode)
    file_path = Path("%s\\data\\timeseries\\%s\\%s\\[%s]timeseries(%s,%s).csv"%(Path(dirname(abspath(__file__))).parent, states[geocode[:2]], inner_folder, geocode, lat, lon))

    try:
        file = open(file_path, "xb")
    except Exception as err:
        print(err)
        con.send(line)
    else: 
        try:
            file.write(request("GET","https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat=%s&lon=%s&components=1" %(lat,lon),preload_content=False,retries=False,timeout=None).data)
        except Exception as err:
            file.close()
            remove(file_path)
            print(err)
            con.send(line)
        else:
            file.close()
            with open(file_path, "r") as f:
                l:str = f.readline()
            if (l.startswith("Latitude (decimal degrees):")):
                if(compressed):
                    compress_file(file_path)
            elif (l.startswith('{"message":"Location over the sea. Please, select another location"') or
                  l.startswith('{"message":"Internal Server Error","status":500}')):
                None
            else:
                remove(file_path)
                con.send(line)
    
    con.close()

def new_coord(coord:str, geocode:str, compressed:bool = True, monolith:bool = False) -> bool:
    inner_folder:str = 'monolith' if monolith else '[%s]'%(geocode)

    file_path:Path = Path("%s\\data\\timeseries\\%s\\%s\\[%s]timeseries(%s).csv"%(
        Path(dirname(abspath(__file__))).parent,
        states[geocode[:2]],
        inner_folder,
        geocode,
        coord
        )
    )
    
    compressed_file_path:Path = Path("%s.gz"%(str(file_path)))

    if(isfile(compressed_file_path)):
       if(getsize(compressed_file_path)>0):
           return False
       else:
           remove(compressed_file_path)
    
    if (isfile(file_path)):
        if(getsize(file_path)>0):
            if(compressed):
                compress_file(file_path)
            return False
        else:
            remove(file_path)
    return True

def gather_to_monolith(state:str) -> None:
    geocodes:list[str] = listdir("%s\\data\\timeseries\\%s"%(Path(dirname(abspath(__file__))).parent, state))
    geocodes.remove('monolith')
    for geocode_folder in geocodes:
        for coord_file in listdir("%s\\data\\timeseries\\%s\\%s"%(Path(dirname(abspath(__file__))).parent, state, geocode_folder)):
            rename("%s\\data\\timeseries\\%s\\%s\\%s"%(Path(dirname(abspath(__file__))).parent, state, geocode_folder, coord_file), '%s\\data\\timeseries\\%s\\monolith\\%s'%(Path(dirname(abspath(__file__))).parent, state, coord_file[9:]))
        remove("%s\\data\\timeseries\\%s\\%s"%(Path(dirname(abspath(__file__))).parent, state, geocode_folder))

# main functions

def city_timeseries(geocode_list:list[str], compressed:bool = True, _rt:bool = False) -> None:
    """ 
    A city_timeseries recebe uma lista de geocódigos(municípos -> 7 dígitos) e baixa os municípios em sequência,
    mas paralelamente todas as coordenadas do município. 
    Opcionais:
        compressed -> booleano que dita se irá comprimir os arquivos ou não. Por padrão, irá comprimir.
        rt -> NÃO atribua nada. 
    """
    for geocode in geocode_list:
        t0:float = perf_counter()
        
        if (not(_rt)):
            timeseries_coords_folder:str = "%s\\data\\timeseries_coords_filtered"%(Path(dirname(abspath(__file__))).parent)
            state_folder:str = '%s\\%s'%(timeseries_coords_folder, [path for path in listdir(timeseries_coords_folder) if path.startswith(states[geocode[:2]])][0])
            coords_path:Path = Path('%s\\%s'%(state_folder, next(f for f in listdir(state_folder) if f[1:8] == geocode)))

            with open(coords_path, "r", 1024**2, 'utf-8') as inputs:
                lines:list[str] = [line[:-1] for line in inputs.readlines() if(new_coord(line[:-1], geocode, compressed))]
            
        else:
            coords_path = Path("%s\\data\\timeseries_coords_filtered\\retry.dat"%(Path(dirname(abspath(__file__))).parent))
            with open(coords_path, "r") as inputs:
                lines = inputs.readlines()
        

        if(lines):
            parent_cons:tuple[PipeConnection]
            child_cons:tuple[PipeConnection]
            parent_cons, child_cons = zip(*[Pipe(False) for _ in range(len(lines))])
            processes:list[Process] = [Process(target=request_timeseries, args=[line, geocode, con, compressed]) for (line, con) in zip(lines, child_cons)]

            makedirs("%s\\data\\timeseries\\%s\\[%s]"%(Path(dirname(abspath(__file__))).parent, states[geocode[:2]], geocode), exist_ok=True)

            sleep_count:int = 0
            i:int = 0
            while(i<len(processes)):
                while(i<len(processes) and len(active_children())<100):
                    while ((virtual_memory()[0]-virtual_memory()[3])/(1024**2)<311):
                        sleep(1)
                        sleep_count += 1
                    
                    processes[i].start()
                    i += 1
                    sleep(0.04)

            for process in processes:
                process.join()
                process.close()
            
            print("  Sleep duration: %.2f"%(0.04*len(processes)+sleep_count))
            print("Request duration: %.2f"%(perf_counter()-t0))


            parent_recvs:list[str] = [con.recv() for con in parent_cons if con.poll()]
            if (parent_recvs):
                retry_path:Path = Path("%s\\data\\timeseries_coords_filtered\\retry.dat"%(Path(dirname(abspath(__file__))).parent))
                with open(retry_path, "w") as retry:
                    retry.writelines(parent_recvs)
                print("Retrying %i coordinates..."%(len(parent_recvs)))
                city_timeseries([geocode], compressed, True)
                if (isfile(retry_path)):
                    remove(retry_path)
            
            if(not(_rt)): print("[%s] execution time: %.2f" %(geocode, perf_counter()-t0))

def state_timeseries(states_list:list[str] = list(states.values()), compressed:bool = True, monolith:bool = False, _rt:bool = False) -> None:
    """ 
    A state_timeseries recebe uma lista de geocódigos (estado -> 2 primeiros dígitos dos municípios) ou siglas.\n
    Opcionais:\n
        compressed -> booleano que dita se irá comprimir os arquivos ou não. Por padrão, irá comprimir.
          monolith -> booleano que define se as coordenas a serem baixadas estão em um único arquivo estadual. Por padrão, procurará por pastas municipais. 
    """
    timeseries_coords_folder:str = "%s\\data\\timeseries_coords_filtered"%(Path(dirname(abspath(__file__))).parent)
    states_folders:list[str] = listdir(timeseries_coords_folder)

    for state in states_list:
        if state not in states.values():
            print(state, "<- inválido")
            continue

        if monolith:
            t0:float = perf_counter()
        
            if (not(_rt)):
                state_coords_path:str = "%s\\data\\timeseries_coords_filtered\\%s\\%s_coords.csv"%(Path(dirname(abspath(__file__))).parent, state, state)

                makedirs('%s\\data\\timeseries\\%s\\monolith'%(Path(dirname(abspath(__file__))).parent, state), exist_ok=True)

                gather_to_monolith(state)
                
                with open(state_coords_path, "r", encoding='utf-8') as fin:
                    lines:list[str] = []
                    for line in fin:
                        spline:list[str] = line[:-1].rsplit(',', 1)
                        if (new_coord(spline[0], spline[1], compressed, monolith)):
                            lines.append(line)
            else:
                coords_path = Path("%s\\data\\timeseries_coords_filtered\\retry.dat"%(Path(dirname(abspath(__file__))).parent))
                with open(coords_path, "r") as inputs:
                    lines = inputs.readlines()

            if(lines):
                parent_cons:tuple[PipeConnection]
                child_cons:tuple[PipeConnection]
                parent_cons, child_cons = zip(*[Pipe(False) for _ in range(len(lines))])
                processes:list[Process] = [Process(target=request_timeseries, args=[*line.split(','), con, compressed, monolith]) for (line, con) in zip(lines, child_cons)]

                sleep_count:int = 0
                i:int = 0
                while(i<len(processes)):
                    while(i<len(processes) and len(active_children())<100):
                        while ((virtual_memory()[0]-virtual_memory()[3])/(1024**2)<311):
                            sleep(1)
                            sleep_count += 1
                        
                        processes[i].start()
                        i += 1
                        sleep(0.04)

                for process in processes:
                    process.join()
                    process.close()
                
                print("  Sleep duration: %.2f"%(0.04*len(processes)+sleep_count))
                print("Request duration: %.2f"%(perf_counter()-t0))


                parent_recvs:list[str] = [con.recv() for con in parent_cons if con.poll()]
                if (parent_recvs):
                    retry_path:Path = Path("%s\\data\\timeseries_coords_filtered\\retry.dat"%(Path(dirname(abspath(__file__))).parent))
                    with open(retry_path, "w") as retry:
                        retry.writelines(parent_recvs)
                    print("Retrying %i coordinates..."%(len(parent_recvs)))
                    state_timeseries([state], compressed, True, True)
                    if (isfile(retry_path)):
                        remove(retry_path)
                
            if(not(_rt)): print("%s execution time: %.2fmin" %(state, (perf_counter()-t0)/60))
        else:
            t0 = perf_counter()
            state_folder = [path for path in states_folders if path.startswith(state)][0]
            geocode_list:list[str] = [file[1:8] for file in listdir('%s\\%s'%(timeseries_coords_folder, state_folder))]
            city_timeseries(geocode_list, compressed)
            print("%s execution time: %.2fs" %(state, perf_counter()-t0))

def brasil_timeseries(compressed:bool = True) -> None:
    """
    A brasil_timeseries não recebe argumentos obrigatórios.\n
    Opcionais:\n
        compresse -> booleano que dita se irá comprimir os arquivos ou não. Por padrão, irá comprimir.
    """
    t0:float = perf_counter()
    state_timeseries(compressed=compressed)
    print("Brasil execution time: %.2f" %(perf_counter()-t0))

def main() -> None:
    """ 
    Para raio de 1,35 km -> 1.123.128 coordenadas.
        
        Normal: 7,401 TiB (8,138 TB). Obs: Talvez um pouco menos, desconsiderando coordenadas que caíram no mar e/ou que os dados não vão até 2023
    Compactado: 1,451 TiB (1,595 TB). Obs: Compactado com gzip.

    Caso mude o raio, não mexer no começo do nome das pastas e não mexer no nome dos arquivos. 
    """
    #city_timeseries(['3501905'])

    state_timeseries(["SP"], monolith=True)

    #brasil_timeseries()

if __name__ == '__main__':
    main()
    """ monolith_folder:Path = Path('C:\\Programas\\timeseries\\SP\\monolith')
    with open(Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries_coords_filtered\\SP\\SP_coords.csv'), 'r', 1024*1024, 'utf-8') as fin:
        coord_dict:dict[str, str] = {line.rsplit(',', 1)[0]:line.rsplit(',', 1)[1][:-1] for line in fin}

    i:int = 0
    for file in listdir(monolith_folder):
        try: rename(monolith_folder.joinpath(file), '%s\\[%s]%s'%(monolith_folder, coord_dict[file[-29:-8]], file))
        except KeyError:
            print(file)
            i += 1
    print(i) """