from sympy.physics.quantum.cg import CG
from sympy import S
import numpy as np
import re
# 注意 S(3)/2 就表示 3/2， 不计算为 1.5
# CG(j1, m1, j2, m2, J, M)


def vec_pos(j1, j2, m1, m2):
    id1 = (j1 - m1)
    id2 = (j2 - m2)
    return int(id1 * (2 * j2 + 1) + id2)


def J_pos(Jmax, Ji, Mi):
    '''
    Ji in range(Jmin, Jmin+1, ..., Jmax)
    '''
    return int((Ji - Mi) + ((2 * Jmax + 1) + (2 * (Ji+1) + 1)) * (Jmax - Ji) / 2)


def uni_array(start, end, step):
    return np.arange(start, end + step, step)


def CGtrans(j1, j2):

    mat = np.zeros((2 * j1 + 1) * (2 * j2 + 1))
    for J in uni_array(j1 + j2, j1 - j2, -1):
        for M in uni_array(J, -J, -1):
            for m1 in uni_array(j1, -j1, -1):
                m2 = M - m1
                if (m2 > j2 or m2 < - j2):
                    continue
                cg = CG(S(j1 * 2) / 2, S(int(m1 * 2)) /
                        2, S(j2 * 2) / 2, S(m2 * 2) / 2, S(J * 2) / 2, S(M * 2) / 2)
                val = cg.doit()
                match = re.match
                mat[J_pos(j1 + j2, J, M)][vec_pos(j1, m1, j2, m2)]
