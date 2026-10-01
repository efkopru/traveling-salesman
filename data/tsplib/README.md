# TSPLIB instances

Symmetric TSP instances from TSPLIB (G. Reinelt, Universität Heidelberg),
included for benchmarking against published optimal tour lengths.

| Instance | Cities | Distance | Optimal tour length |
|----------|--------|----------|---------------------|
| eil51    | 51     | EUC_2D   | 426                 |
| berlin52 | 52     | EUC_2D   | 7542                |
| st70     | 70     | EUC_2D   | 675                 |
| kroA100  | 100    | EUC_2D   | 21282               |

EUC_2D distances are Euclidean distances rounded to the nearest integer;
`TSPSolver.from_tsplib()` applies this automatically.

Source: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
