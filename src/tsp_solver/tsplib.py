"""Reading TSPLIB instance files."""

from typing import Dict

import numpy as np

# Published optimal tour lengths for the instances bundled in data/tsplib.
TSPLIB_OPTIMA = {
    'eil51': 426,
    'berlin52': 7542,
    'st70': 675,
    'kroA100': 21282,
}


def load_tsplib(path: str) -> Dict:
    """
    Read a symmetric TSPLIB .tsp file with node coordinates.

    Supported EDGE_WEIGHT_TYPEs: EUC_2D, CEIL_2D, ATT.

    Returns:
        Dict with 'name', 'comment', 'dimension', 'edge_weight_type',
        'coordinates' (array of shape (n, 2)), 'city_names' (node ids as
        strings) and 'optimum' (from TSPLIB_OPTIMA, or None).
    """
    header = {}
    ids, coords = [], []
    in_coords = False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == 'EOF':
                break
            if in_coords:
                parts = line.split()
                if len(parts) < 3 or not parts[0].lstrip('-').isdigit():
                    in_coords = False      # start of another section
                else:
                    ids.append(parts[0])
                    coords.append((float(parts[1]), float(parts[2])))
                    continue
            if line.startswith('NODE_COORD_SECTION'):
                in_coords = True
            elif ':' in line:
                key, value = line.split(':', 1)
                header[key.strip().upper()] = value.strip()

    if header.get('TYPE', 'TSP').split()[0] != 'TSP':
        raise ValueError(f"Only symmetric TSP files are supported "
                         f"(TYPE: {header.get('TYPE')})")
    edge_type = header.get('EDGE_WEIGHT_TYPE', '')
    if edge_type not in ('EUC_2D', 'CEIL_2D', 'ATT'):
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE '{edge_type}'; "
                         f"supported: EUC_2D, CEIL_2D, ATT")
    dimension = int(header.get('DIMENSION', len(coords)))
    if len(coords) != dimension:
        raise ValueError(f"DIMENSION is {dimension} but {len(coords)} "
                         f"coordinates were read")

    name = header.get('NAME', '')
    return {
        'name': name,
        'comment': header.get('COMMENT', ''),
        'dimension': dimension,
        'edge_weight_type': edge_type,
        'coordinates': np.array(coords),
        'city_names': ids,
        'optimum': TSPLIB_OPTIMA.get(name),
    }
