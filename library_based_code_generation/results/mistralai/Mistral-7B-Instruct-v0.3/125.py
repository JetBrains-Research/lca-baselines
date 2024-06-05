 ```python
from pyscf import gto, scf, cc, df
import numpy as np

cell = gto.Cell()
cell.atom = '''
    O   0.0   0.0   0.0
    H   0.0   0.0   a_bohr
    H   0.0   0.0   -a_bohr
    H    a_bohr   0.0   0.0
    H   -a_bohr  0.0   0.0
    H   0.0   a_bohr   0.0
    H   0.0   -a_bohr  0.0
    H   0.0   0.0   a_bohr/sqrt(2)
    H   0.0   0.0   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   0.0   0.0
    H   -a_bohr/sqrt(2)  0.0   0.0
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   0.0   -a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)  0.0   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   0.0   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   0.0   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   0.0   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   0.0   -a_bohr/sqrt(2)  -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)
    H   -a_bohr/sqrt(2)   -a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)
    H   0.0   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   a_bohr/sqrt(2)   -a_bohr/sqrt(2)   -a