import numpy as np
from CG_coef_calculation.simple_schur_transform import CG_Coef
# from CG_coef_calculation.CG_transform_sympy import CG_Coef
from scipy.linalg import block_diag
import time


def uni_array(begin, end, step):
    '''
    return an array contain [begin...end] by step
    '''
    return np.arange(begin, end+step, step)


class J_invariant_space:

    def __init__(self, J, N):
        self.J = J
        self.N = N

    def next(self, J_merge):
        ret = list()
        for J1 in self.J:
            for J2 in J_merge.J:
                for new_J in uni_array(J1 + J2, np.abs(J1 - J2), -1):
                    ret.append(new_J)
        return J_invariant_space(ret, self.N + J_merge.N)


def merge(J1s, J2s, Num_of_identity_to_pre_tensor, Num_of_identity_to_suf_tensor):
    '''
    把 J1s tensor J2s 这个空间分解为 J invariance 子空间直和形式

    每个 J_id 中对应的 M_id 是下标从小到大 M值从大到小
    '''
    mats = list()

    J1_row = 0
    for J1 in J1s.J:
        J2_row = 0
        for J2 in J2s.J:
            pre_mat = CG_Coef(J1, J2)
            new_mat = np.zeros([pre_mat.shape[0], 2**(J1s.N + J2s.N)])
            for J_and_M in range(pre_mat.shape[0]):
                for M1 in uni_array(J1, -J1, -1):
                    for M2 in uni_array(J2, -J2, -1):
                        # 在CG空间中(J1, M1) tensor (J2, M2)的编号
                        id_pre = int((J1 - M1) * (2 * J2 + 1) + (J2 - M2))
                        if np.abs(pre_mat[J_and_M][id_pre]) < 1e-7:  # 如果这个格子没值 不考虑
                            continue
                        # 在第一个子空间中(J1, M1)的编号
                        idm1_in1 = int(J1_row + (J1 - M1))
                        # 在第二个子空间中(J2, M2)的编号
                        idm2_in2 = int(J2_row + (J2 - M2))
                        id_suf = idm1_in1 * (2**(J2s.N)) + \
                            idm2_in2  # 在总空间中tensor的编号
                        new_mat[J_and_M][id_suf] = pre_mat[J_and_M][id_pre]
            J2_row += int(2 * J2 + 1)
            mats.append(new_mat)
        J1_row += int(2 * J1 + 1)

    ret = np.vstack(mats)

    for i in range(Num_of_identity_to_pre_tensor):
        ret = np.kron(np.array([[1, 0], [0, 1]]), ret)

    for i in range(Num_of_identity_to_suf_tensor):
        ret = np.kron(ret, np.array([[1, 0], [0, 1]]))
    return ret


def merge_spin_one_by_one(n):
    single_spin = J_invariant_space([1/2], 1)
    J_fixed = single_spin

    mat = np.eye(2**n, 2**n)
    for i in range(n-1):
        mat = np.dot(merge(J_fixed, single_spin, 0, n-i-2), mat)
        J_fixed = J_fixed.next(single_spin)

    # print(mat.round(decimals=2))
    # np.savetxt("one_by_one_result.txt", mat, fmt="%.2lf", delimiter="\t")


def merge_spin_divide_and_conquer(n):
    J_fixed = [J_invariant_space([1/2], 1) for i in range(n)]
    mat = np.eye(2**n, 2**n)

    while (len(J_fixed) != 1):
        num_J_need_merge = len(J_fixed)

        pre_qubits = 0
        suf_qubits = n

        next_J_fixed = []

        for i in range(0, num_J_need_merge-1, 2):  # merge (Ji, Ji+1) per try
            next_J_fixed.append(J_fixed[i].next(J_fixed[i+1]))

            this_qubits = J_fixed[i].N + J_fixed[i+1].N
            suf_qubits -= this_qubits

            mat = np.dot(
                merge(J_fixed[i], J_fixed[i+1], pre_qubits, suf_qubits), mat)

            pre_qubits += this_qubits

        if num_J_need_merge % 2 != 0:
            next_J_fixed.append(J_fixed[-1])

        J_fixed = next_J_fixed

    # print(mat.round(decimals=2))
    # np.savetxt("divide_and_conquer_result.txt", mat, fmt="%.2lf", delimiter="\t")


def merge_special(n):
    J_fixed = [J_invariant_space([1/2], 1) for i in range(n)]
    mat = np.eye(2**n, 2**n)

    J2 = J_fixed[1].next(J_fixed[2])
    mat = np.dot(merge(J_fixed[1], J_fixed[2], 1, 0), mat)

    mat = np.dot(merge(J_fixed[0], J2, 0, 0), mat)
    np.savetxt("divide_and_conquer_special_result.txt",
               mat, fmt="%.2lf", delimiter="\t")


if __name__ == "__main__":
    n = 12
    start_time = time.time()
    # merge_spin_one_by_one(n)
    merge_spin_divide_and_conquer(n)
    end_time = time.time()
    print(end_time - start_time)
    # merge_special(n)
