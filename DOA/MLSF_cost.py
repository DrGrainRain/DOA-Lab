import numpy as np

"""
=============================================
统一接口说明（所有代价函数共用这套输入参数）
=============================================
输入参数统一含义：
    A         : 阵列流型矩阵（导向矢量矩阵），维度 [阵元数 M, 信源数 K]
    theta_deg : 入射波DOA角度（单位：度），维度 [K,]，仅作为接口兼容，实际由阵列流型矩阵计算
    data      : 核心输入数据（根据代价函数自动适配）
                - DML/SML：传入样本协方差矩阵 Rhat
                - SSF/WSF：传入信号子空间 U_S
                - NSF/WNSF：传入噪声子空间 U_N
                - GWSF：传入数据矩阵 M_mat
    W         : 加权矩阵
                - 无需加权的代价函数（DML/SML/SSF/NSF）：传入 None 或任意值即可，函数内部不使用
                - WSF/WNSF/GWSF：传入对应最优加权矩阵
    M         : 阵列阵元数量
    d         : 阵元间距（通常为半波长 λ/2）
    lambd     : 信号波长
输出：
    J         : 代价值（实数，越小表示拟合效果越好）
=============================================
"""

def dml_cost(A, theta_deg, data, W, M, d, lambd):
    """DML 确定性最大似然代价函数"""
    Rhat = data
    P_A = A @ np.linalg.pinv(A)
    P_A_perp = np.eye(M) - P_A
    J = np.real(np.trace(P_A_perp @ Rhat))
    return J

def sml_cost1(A, theta_deg, data, W, M, d, lambd):
    """SML 随机最大似然代价函数 1（协方差矩阵对数行列式形式）"""
    Rhat = data
    K = A.shape[1]
    A_pinv = np.linalg.pinv(A)
    P_A = A @ A_pinv
    P_A_perp = np.eye(M) - P_A
    sigma2_sml = np.real(np.trace(P_A_perp @ Rhat)) / (M - K)
    R_S = A_pinv @ (Rhat - sigma2_sml * np.eye(M)) @ A_pinv.conj().T
    cov_matrix = A @ R_S @ A.conj().T + sigma2_sml * np.eye(M)
    sign, logdet = np.linalg.slogdet(cov_matrix)
    J = logdet
    return np.real(J)

def sml_cost2(A, theta_deg, data, W, M, d, lambd):
    """SML 随机最大似然代价函数 2（简化对数似然形式）"""
    Rhat = data
    K = A.shape[1]
    A_pinv = np.linalg.pinv(A)
    P_A = A @ A_pinv
    P_A_perp = np.eye(M) - P_A
    sigma2_sml = np.real(np.trace(P_A_perp @ Rhat)) / (M - K)
    term1 = 2 * (M - K) * np.log(sigma2_sml)
    A_RA = A_pinv @ Rhat @ A
    sign, logdet = np.linalg.slogdet(A_RA)
    term2 = logdet
    J = term1 + term2
    return np.real(J)

def ssf_cost(A, theta_deg, data, W, M, d, lambd):
    """SSF 信号子空间拟合代价函数"""
    U_S = data
    A_pinv = np.linalg.pinv(A)
    P_A = A @ A_pinv
    P_A_perp = np.eye(M) - P_A
    U_S_outer = U_S @ U_S.conj().T
    J = np.real(np.trace(P_A_perp @ U_S_outer))
    return J

def wsf_cost(A, theta_deg, data, W, M, d, lambd):
    """WSF 加权信号子空间拟合代价函数"""
    U_S = data
    A_pinv = np.linalg.pinv(A)
    P_A = A @ A_pinv
    P_A_perp = np.eye(M) - P_A
    U_W_UH = U_S @ W @ U_S.conj().T
    J = np.real(np.trace(P_A_perp @ U_W_UH))
    return J

def nsf_cost(A, theta_deg, data, W, M, d, lambd):
    """NSF 噪声子空间拟合代价函数"""
    U_N = data
    AH_UN = A.conj().T @ U_N
    J = np.real(np.trace(AH_UN @ AH_UN.conj().T))
    return J

def wnsf_cost(A, theta_deg, data, W, M, d, lambd):
    """WNSF 加权噪声子空间拟合代价函数"""
    U_N = data
    AH_UN = A.conj().T @ U_N
    term = AH_UN @ W @ AH_UN.conj().T
    J = np.real(np.trace(term))
    return J

def gwsf_cost(A, theta_deg, data, W, M, d, lambd):
    """GWSF 通用加权子空间拟合代价函数"""
    M_mat = data
    A_pinv = np.linalg.pinv(A)
    P_A = A @ A_pinv
    P_A_perp = np.eye(M) - P_A
    M_W_MH = M_mat @ W @ M_mat.conj().T
    J = np.real(np.trace(P_A_perp @ M_W_MH))
    return J


# ===================== 仿真参数（已修复维度）=====================
M = 8          # 阵元数
K = 2          # 信源数
d = 0.5        # 阵元间距
lambd = 1.0    # 波长
theta_deg = np.array([10, 30])

A = np.random.randn(M, K) + 1j * np.random.randn(M, K)
Rhat = np.random.randn(M, M) + 1j * np.random.randn(M, M)
Rhat = Rhat @ Rhat.conj().T
U_S = np.random.randn(M, K) + 1j * np.random.randn(M, K)
U_N = np.random.randn(M, M-K) + 1j * np.random.randn(M, M-K)

# 加权矩阵（已修复）
W = np.eye(K)          # WSF 用：K×K
W_N = np.eye(M-K)      # WNSF 用：(M-K)×(M-K) ✔✔✔
M_mat = U_S

# ===================== 全部调用（无报错）=====================
J_dml = dml_cost(A, theta_deg, Rhat, None, M, d, lambd)
J_sml1 = sml_cost1(A, theta_deg, Rhat, None, M, d, lambd)
J_sml2 = sml_cost2(A, theta_deg, Rhat, None, M, d, lambd)
J_ssf = ssf_cost(A, theta_deg, U_S, None, M, d, lambd)
J_wsf = wsf_cost(A, theta_deg, U_S, W, M, d, lambd)
J_nsf = nsf_cost(A, theta_deg, U_N, None, M, d, lambd)
J_wnsf = wnsf_cost(A, theta_deg, U_N, W_N, M, d, lambd)
J_gwsf = gwsf_cost(A, theta_deg, M_mat, W, M, d, lambd)

print("J_dml   =", J_dml)
print("J_sml1  =", J_sml1)
print("J_sml2  =", J_sml2)
print("J_ssf   =", J_ssf)
print("J_wsf   =", J_wsf)
print("J_nsf   =", J_nsf)
print("J_wnsf  =", J_wnsf)
print("J_gwsf  =", J_gwsf)