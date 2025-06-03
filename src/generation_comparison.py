import numpy as np
from time import perf_counter
from collections import defaultdict


from usine_generation import usine_plot, usines_plot, usines_pv_mmd_generation
from venture_generation import ventures_process, save_ventures_timeseries_coords_filtered, plot_generation

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def main() -> None:

    #Ventures build
    t0:float = perf_counter()
    states_cities_coords_array:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = ventures_process()
    print('Ventures process execution time:', perf_counter()-t0)
    
    #states_cities_coords_array comprehension
    #print(states_cities_coords_array.keys())
    #print(states_cities_coords_array['SP'].keys())
    #print(sorted([(coord, array.shape[0]) for coord, array in states_cities_coords_array['SP']['3541406'].items()], key=lambda e: e[1])[:-5])
    #print(states_cities_coords_array['SP']['3541406']['(-21.899087,-51.288853)'])
    
    #biggest sp coord: 3502101 (-20.905187,-51.370022) 908

    #Debugger
    #print(*((coord, array.shape[0]) for coord, array in states_cities_coords_array['SP']['3541406'].items()), sep='\n')
    
    #Save filtered coords
    """ t0 = perf_counter()
    save_ventures_timeseries_coords_filtered(states_cities_coords_array, True)
    print('Save filtered timeseries coords time:', perf_counter()-t0) """

    #Ventures timeseries data volume analysis
    """ total:int = 0
    for state, cities in states_cities_coords_array.items():
        subtotal:int = sum([len(city.keys()) for city in cities.values()])
        total += subtotal
        print('%s: %5i timeseries coords (%5.2f GiB compressed -> %5.1fmin)'%(state, subtotal, 1650*subtotal/(1024**2), 0.42981*subtotal/60))
    print('Total space required: %i timeseries (%.2f GiB after compression)'%(total, 1650*total/(1024**2)))
    print('Total minimum expected download time: Uncompressed %.2f TiB -> %.1fh'%(7.65*total/(1024**2), 0.42981*total/(60**2))) """
    
    #Usines build
    t0 = perf_counter()
    per_state_usines_pv_mmd_generation:dict[str, np.ndarray] = usines_pv_mmd_generation()
    print('Usines process execution time:', perf_counter()-t0)

    ##################################################################################################
    print('')
    
    period:tuple[int, int] = (20230429, 20241231)
    #period:tuple[int, int] = (20240000+m*100, 20240000+m*100+99)

    #Ventures plot // depends on having timeseries
    t0 = perf_counter()
    plot_generation(states_cities_coords_array['SP'], period, monolith=True)
    print('Ventures generetaion plot execution time:', perf_counter()-t0)
    
    #Usines plot
    t0 = perf_counter()
    usine_plot('SP', per_state_usines_pv_mmd_generation['SP'], period)
    print('Usines generation plot execution time:', perf_counter()-t0)

if __name__ == "__main__":
    main()

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')