"""Reading TSPLIB instance files."""

import os
from typing import Dict, Union

import numpy as np

# Published optimal tour lengths, by instance name, of the TSPLIB instances
# this module can read (EUC_2D and ATT distances, up to about 6000 cities).
# Source: "Optimal solutions for symmetric TSPs", TSPLIB95,
# http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html
TSPLIB_OPTIMA = {
    'a280': 2579, 'att48': 10628, 'att532': 27686, 'berlin52': 7542,
    'bier127': 118282, 'ch130': 6110, 'ch150': 6528, 'd198': 15780,
    'd493': 35002, 'd657': 48912, 'd1291': 50801, 'd1655': 62128,
    'd2103': 80450, 'eil51': 426, 'eil76': 538, 'eil101': 629,
    'fl417': 11861, 'fl1400': 20127, 'fl1577': 22249, 'fl3795': 28772,
    'fnl4461': 182566, 'gil262': 2378, 'kroA100': 21282, 'kroB100': 22141,
    'kroC100': 20749, 'kroD100': 21294, 'kroE100': 22068, 'kroA150': 26524,
    'kroB150': 26130, 'kroA200': 29368, 'kroB200': 29437, 'lin105': 14379,
    'lin318': 42029, 'nrw1379': 56638, 'p654': 34643, 'pcb442': 50778,
    'pcb1173': 56892, 'pcb3038': 137694, 'pr76': 108159, 'pr107': 44303,
    'pr124': 59030, 'pr136': 96772, 'pr144': 58537, 'pr152': 73682,
    'pr226': 80369, 'pr264': 49135, 'pr299': 48191, 'pr439': 107217,
    'pr1002': 259045, 'pr2392': 378032, 'rat99': 1211, 'rat195': 2323,
    'rat575': 6773, 'rat783': 8806, 'rd100': 7910, 'rd400': 15281,
    'rl1304': 252948, 'rl1323': 270199, 'rl1889': 316536, 'rl5915': 565530,
    'rl5934': 556045, 'st70': 675, 'ts225': 126643, 'tsp225': 3916,
    'u159': 42080, 'u574': 36905, 'u724': 41910, 'u1060': 224094,
    'u1432': 152970, 'u1817': 57201, 'u2152': 64253, 'u2319': 234256,
    'vm1084': 239297, 'vm1748': 336556,
}

SUPPORTED_EDGE_WEIGHT_TYPES = ('EUC_2D', 'CEIL_2D', 'ATT')


def load_tsplib(path: Union[str, os.PathLike]) -> Dict:
    """
    Read a symmetric TSPLIB .tsp file with node coordinates.

    Supported EDGE_WEIGHT_TYPEs: EUC_2D, CEIL_2D, ATT.

    Returns:
        Dict with 'name', 'comment', 'dimension', 'edge_weight_type',
        'coordinates' (array of shape (n, 2)), 'city_names' (node ids as
        strings) and 'optimum' (the published optimal tour length if the
        file's NAME is in TSPLIB_OPTIMA, else None).
    """
    header = {}
    ids, coords = [], []
    in_coords = False
    # The format is ASCII; a stray byte in a comment should not stop the read.
    with open(path, encoding='utf-8', errors='replace') as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            keyword = line.upper()
            if keyword == 'EOF':
                break
            if in_coords:
                parts = line.split()
                try:
                    x, y = float(parts[1]), float(parts[2])
                except (ValueError, IndexError):
                    if not line[0].isalpha():
                        raise ValueError(f"line {line_number}: expected 'id x y', "
                                         f"got '{line}'") from None
                    in_coords = False      # a keyword: another section starts
                else:
                    ids.append(parts[0])
                    coords.append((x, y))
                    continue
            if keyword.startswith('NODE_COORD_SECTION'):
                in_coords = True
            elif ':' in line:
                key, value = line.split(':', 1)
                header[key.strip().upper()] = value.strip()

    problem_type = header.get('TYPE') or 'TSP'
    if problem_type.split()[0].upper() != 'TSP':
        raise ValueError(f"Only symmetric TSP files are supported "
                         f"(TYPE: {problem_type})")
    edge_type = header.get('EDGE_WEIGHT_TYPE', '').upper()
    if edge_type not in SUPPORTED_EDGE_WEIGHT_TYPES:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE '{edge_type}'; "
                         f"supported: {', '.join(SUPPORTED_EDGE_WEIGHT_TYPES)}")
    if not coords:
        raise ValueError("no node coordinates found (NODE_COORD_SECTION "
                         "is missing or empty)")
    if 'DIMENSION' in header:
        try:
            dimension = int(header['DIMENSION'])
        except ValueError:
            raise ValueError(f"DIMENSION must be an integer "
                             f"(got '{header['DIMENSION']}')") from None
        if len(coords) != dimension:
            raise ValueError(f"DIMENSION is {dimension} but {len(coords)} "
                             f"coordinates were read")
    dimension = len(coords)

    name = header.get('NAME', '')
    return {
        'name': name,
        'comment': header.get('COMMENT', ''),
        'dimension': dimension,
        'edge_weight_type': edge_type,
        'coordinates': np.array(coords),
        'city_names': ids,
        # Some files give the name with its extension ("NAME: eil51.tsp").
        'optimum': TSPLIB_OPTIMA.get(name[:-4] if name.lower().endswith('.tsp')
                                     else name),
    }
