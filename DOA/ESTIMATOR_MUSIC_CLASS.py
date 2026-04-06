import numpy as np
import matplotlib.pyplot as plt

class MUSIC_Estimator:
    def __init__(self, R, M, K):
        self.R = R
        self.M = M
        self.K = K
        self.theta_scan = np.linspace(-90, 90, 1801)

    def MUSIC_spectrum(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        self.eigenvalues_signal = eigenvalues[:self.K]
        self.eigenvalues_noise = eigenvalues[self.K:]
        self.eigenvectors_signal = eigenvectors[:, :self.K]
        self.eigenvectors_noise = eigenvectors[:, self.K:]
        return self.eigenvalues_signal, self.eigenvalues_noise, self.eigenvectors_signal, self.eigenvectors_noise

    def MUSIC_estimator(self):
        P_theta = np.zeros(len(self.theta_scan))
        for i, theta in enumerate(self.theta_scan):
            a_theta = np.exp(1j * np.pi * np.arange(self.M) * np.sin(np.deg2rad(theta)))
            denominator = np.linalg.norm(self.eigenvectors_noise.conj().T @ a_theta) ** 2
            if denominator == 0:
                P_theta[i] = 1e10
            else:
                P_theta[i] = 1 / denominator
        return P_theta
    
    def WMUSIC_Estimator(self, type="MUSIC"):
        P_theta = np.zeros(len(self.theta_scan), dtype=np.float64)
        Un = self.eigenvectors_noise
        R_inv = np.linalg.inv(self.R)

  
        c      = Un[0, :].reshape(-1, 1)
        EN_hat = Un[1:, :]
        ccH    = (c.conj().T @ c).item()  
        d      = np.zeros((self.M, 1), dtype=np.complex128)
        d[0,0] = 1.0
        d[1:]  = EN_hat @ c.conj() / ccH 
        ddH    = d @ d.conj().T

        for i, theta in enumerate(self.theta_scan):
            a_theta = np.exp(1j * np.pi * np.arange(self.M) * np.sin(np.deg2rad(theta))).reshape(-1, 1)

            if type == "MUSIC":
                den = np.linalg.norm(Un.conj().T @ a_theta) ** 2
                P_theta[i] = 1.0 / (den + 1e-12)

            elif type == "MNM":
                den = a_theta.conj().T @ ddH @ a_theta
                P_theta[i] = 1.0 / (np.abs(den.item()) + 1e-12)

            elif type == "MVM":
                den = a_theta.conj().T @ R_inv @ a_theta
                P_theta[i] = 1.0 / (np.abs(den.item()) + 1e-12)

            elif type == "MEM":
                u0 = np.zeros((self.M, 1), dtype=np.complex128)
                u0[0, 0] = 1.0
                term = R_inv @ u0
                den = a_theta.conj().T @ term @ term.conj().T @ a_theta
                P_theta[i] = 1.0 / (np.abs(den.item()) + 1e-12)

            else:
                print("Invalid type")
                return None            

        return P_theta


    def Beamspace_MUSIC(self, B, T_type='avg_subarray'):
        """
        王永良 这个书P120 只有DFT 和 第三个算法 效果还行,别的算法出图效果很差,需要调整参数或者改进算法细节。 
        """


        M, K = self.M, self.K
        if B <= K:
            raise ValueError("B must be > K")
        R_hat = self.R

        # ---------- 构造波束形成矩阵 T ----------
        if T_type == 'DFT':
            angles = np.linspace(0, 35, B)   # 可改为全角 np.linspace(-90,90,B)
            T = np.zeros((M, B), dtype=complex)
            for b in range(B):
                sin_angle = np.sin(np.deg2rad(angles[b]))
                T[:, b] = np.exp(1j * np.pi * np.arange(M) * sin_angle)
            T = T / np.sqrt(M)

        elif T_type == 'avg_subarray':
            L = M - B + 1
            T = np.zeros((M, B), dtype=complex)
            for i in range(B):
                T[i:i+L, i] = 1.0 / np.sqrt(L)
            Q, _ = np.linalg.qr(T)
            T = Q[:, :B]

        elif T_type == 'block_avg':
            if M % B != 0:
                raise ValueError(f"M ({M}) must be divisible by B ({B}) for block_avg mode.")
            L = M // B
            T = np.zeros((M, B), dtype=complex)
            for i in range(B):
                start = i * L
                T[start:start+L, i] = 1.0 / np.sqrt(L)
            Q, _ = np.linalg.qr(T)
            T = Q[:, :B]

        elif T_type == 'beam_steering':
            theta_left, theta_right = -60, 60   # 可参数化
            theta_steer = np.linspace(theta_left, theta_right, B)
            C = np.zeros((M, B), dtype=complex)
            for i, th in enumerate(theta_steer):
                C[:, i] = np.exp(1j * np.pi * np.arange(M) * np.sin(np.deg2rad(th)))
            U, _, _ = np.linalg.svd(C, full_matrices=False)
            T = U[:, :B]

        elif T_type == 'optimal':
            theta_int = np.linspace(-90, 90, 1000)
            d_theta = np.deg2rad(theta_int[1] - theta_int[0])
            Q = np.zeros((M, M), dtype=complex)
            for th in theta_int:
                a = np.exp(1j * np.pi * np.arange(M) * np.sin(np.deg2rad(th))).reshape(-1, 1)
                Q += a @ a.conj().T * d_theta
            eigvals, eigvecs = np.linalg.eigh(Q)
            idx = np.argsort(eigvals)[::-1]
            T = eigvecs[:, idx[:B]]

        else:
            raise ValueError("Invalid T_type")

        # ---------- 波束空间变换 ----------
        R_yy = T.conj().T @ R_hat @ T
        # ---------- 波束空间 MUSIC 谱估计 ----------
        eigvals, eigvecs = np.linalg.eigh(R_yy)
        idx = np.argsort(eigvals)[::-1]
        Un_beam = eigvecs[:, idx[K:]]
        P_theta = np.zeros(len(self.theta_scan))
        for i, theta in enumerate(self.theta_scan):
            a_theta = np.exp(1j * np.pi * np.arange(M) * np.sin(np.deg2rad(theta))).reshape(-1, 1)
            a_beam = T.conj().T @ a_theta
            den = np.linalg.norm(Un_beam.conj().T @ a_beam) ** 2
            P_theta[i] = 1.0 / (den + 1e-12)
        return P_theta
    
    def Matrix_reconstruction_MUSIC(self, m,mode="ESVD"):
        M = self.M
        K = self.K
        
        if mode == "ESVD":
            eig_vals, eig_vecs = np.linalg.eigh(self.R)
            idx = np.argsort(eig_vals)[::-1]
            e1 = eig_vecs[:, idx[0]].reshape(-1, 1)

            p = M - m + 1
            Y = np.zeros((m, p), dtype=np.complex128)
            for i in range(m):
                for j in range(p):
                    Y[i, j] = e1[i + j, 0]
            U, _, _ = np.linalg.svd(Y)
            Un = U[:, K:]
            P_theta = np.zeros_like(self.theta_scan, dtype=np.float64)
            for i, th in enumerate(self.theta_scan):
                a = np.exp(1j * np.pi * np.arange(m) * np.sin(np.deg2rad(th))).reshape(-1, 1)
                den = np.linalg.norm(Un.conj().T @ a) ** 2
                P_theta[i] = 1.0 / (den + 1e-12)
            
        elif mode == "Toeplitz":
            R_tp = np.zeros((M, M), dtype=np.complex128)
            for d in range(M):
                avg = np.diag(self.R, d).mean()
                np.fill_diagonal(R_tp[:, d:], avg)
                if d > 0:
                    np.fill_diagonal(R_tp[d:, :], avg.conj())

            eig_vals, eig_vecs = np.linalg.eigh(R_tp)
            idx = np.argsort(eig_vals)[::-1]
            Un = eig_vecs[:, idx[K:]]
            P_theta = np.zeros_like(self.theta_scan, dtype=np.float64)
            for i, th in enumerate(self.theta_scan):
                a = np.exp(1j * np.pi * np.arange(M) * np.sin(np.deg2rad(th))).reshape(-1, 1)
                den = np.linalg.norm(Un.conj().T @ a) ** 2
                P_theta[i] = 1.0 / (den + 1e-12)
        elif mode == "Matrix_Decomp":
            R_f = np.zeros((m, m), dtype=np.complex128)
            p = M - m + 1
            for k in range(p):
                sl = slice(k, k + m)
                R_f += self.R[sl, :][:, sl]
            R_f /= p

            eig_vals, eig_vecs = np.linalg.eigh(R_f)
            idx = np.argsort(eig_vals)[::-1]
            Un = eig_vecs[:, idx[K:]]

            P_theta = np.zeros_like(self.theta_scan, dtype=np.float64)
            for i, th in enumerate(self.theta_scan):
                a = np.exp(1j * np.pi * np.arange(m) * np.sin(np.deg2rad(th))).reshape(-1, 1)
                den = np.linalg.norm(Un.conj().T @ a) ** 2
                P_theta[i] = 1.0 / (den + 1e-12)
        else:
            raise ValueError("Invalid m for Matrix_reconstruction_MUSIC. Use 'ESVD' or 'Toeplitz' or 'Matrix_Decomp'.")
        
        return P_theta



    def spatial_smoothing_MUSIC(self, m, mode='forward'):
 
        M = self.M
        K = self.K
        p = M - m + 1
        R_hat = self.R
        J = np.fliplr(np.eye(m, dtype=np.complex128))

        def build_Z(k):
            left = np.zeros((m, k-1))
            mid = np.eye(m)
            right = np.zeros((m, p - k))
            return np.hstack([left, mid, right])

        def build_Q(k):
            left = np.zeros((m, k-1))
            mid = J
            right = np.zeros((m, p - k))
            return np.hstack([left, mid, right])


        if mode == 'forward':
            Rf = 0
            for k in range(1, p+1):
                Zk = build_Z(k)
                Rf += Zk @ R_hat @ Zk.conj().T
            R_smooth = Rf / p


        elif mode == 'backward':
            Rb = 0
            for k in range(1, p+1):
                Qk = build_Q(k)
                Rb += Qk @ R_hat.conj() @ Qk.conj().T
            R_smooth = Rb / p

        elif mode == 'bidirectional':
            Rf = 0
            for k in range(1, p+1):
                Zk = build_Z(k)
                Rf += Zk @ R_hat @ Zk.conj().T
            Rf /= p

            Rb = 0
            for k in range(1, p+1):
                Qk = build_Q(k)
                Rb += Qk @ R_hat.conj() @ Qk.conj().T
            Rb /= p

            R_smooth = 0.5 * (Rf + Rb)

        else:
            raise ValueError("mode error")

        # ===================== MUSIC 谱 =====================
        eigvals, eigvecs = np.linalg.eigh(R_smooth)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        Un = eigvecs[:, K:]

        P_theta = np.zeros_like(self.theta_scan)
        for i, th in enumerate(self.theta_scan):
            a = np.exp(1j * np.pi * np.arange(m) * np.sin(np.deg2rad(th))).reshape(-1,1)
            den = np.linalg.norm(Un.conj().T @ a) ** 2
            P_theta[i] = 1/(den + 1e-12)

        return P_theta


    def figure_plot(self, spectrum_dict, true_angles): 
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix' 
        plt.figure(figsize=(10, 6))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'olive', 'pink', 'gray']
        for i, (name, P) in enumerate(spectrum_dict.items()):
            plt.plot(
                self.theta_scan,
                10 * np.log10(P / np.max(P)),
                label=f'{name} Spectrum',
                linewidth=2,
                color=colors[i % len(colors)]
            )

        for ang in true_angles:
            plt.axvline(x=ang, color='red', linestyle='--', linewidth=1.5)

        plt.title('DOA Estimation Comparison', fontsize=14, weight='normal')
        plt.xlabel('Angle (degrees)', fontsize=12)
        plt.ylabel('Normalized Spectrum (dB)', fontsize=12)
        plt.xlim(-90, 90)
        plt.ylim(-50, 0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.show()

    def find_peaks(self, P_theta):
        from scipy.signal import find_peaks
        spectrum_db = 10 * np.log10(P_theta / np.max(P_theta) + 1e-10)
  
        peaks, properties = find_peaks(spectrum_db, height=-40, distance=5)
        
        
        peak_heights = spectrum_db[peaks]
        sorted_idx = np.argsort(peak_heights)[::-1]  
        sorted_peaks = peaks[sorted_idx]

        selected_peaks = sorted_peaks[:self.K]
        
        selected_angles = np.sort(self.theta_scan[selected_peaks])
        
        return selected_angles



# ========mian=========
if __name__ == "__main__":
    import sys
    sys.path.append("E:/Code/A_YBTOOL")
    from DOA.gensignal import ArraySimulator

    # 1. 生成信号
    sim = ArraySimulator(M=8, degsais=[12, 25, 33], sigma2=2, SNR_db=10, N=200)
    xmt, R, A, S = sim.generate_data(noise_type="uniform", signal_type="incoherent")  # 相干信号更适合平滑类算法

    music = MUSIC_Estimator(R, M=sim.M, K=sim.K)
    music.MUSIC_spectrum()

    # ===================== 算法 1：标准 & 加权 MUSIC =====================
    P_music = music.WMUSIC_Estimator(type="MUSIC")
    P_mnm   = music.WMUSIC_Estimator(type="MNM")
    P_mvm   = music.WMUSIC_Estimator(type="MVM")
    P_mem   = music.WMUSIC_Estimator(type="MEM")

    # ===================== 算法 2：波束空间 MUSIC =====================
    P_beam_dft = music.Beamspace_MUSIC(B=5, T_type='DFT')
    P_beam_avg = music.Beamspace_MUSIC(B=5, T_type='avg_subarray')

    # ===================== 算法 3：矩阵重构 MUSIC =====================
    m_sub = 6  # 子阵大小
    P_esvd  = music.Matrix_reconstruction_MUSIC(m=m_sub, mode="ESVD")
    P_toep  = music.Matrix_reconstruction_MUSIC(m=m_sub, mode="Toeplitz")
    P_decomp = music.Matrix_reconstruction_MUSIC(m=m_sub, mode="Matrix_Decomp")

    # ===================== 算法 4：空间平滑 MUSIC =====================
    P_ss_f = music.spatial_smoothing_MUSIC(m=m_sub, mode='forward')
    P_ss_b = music.spatial_smoothing_MUSIC(m=m_sub, mode='backward')
    P_ss_bi = music.spatial_smoothing_MUSIC(m=m_sub, mode='bidirectional')

    # ===================== 把所有谱放进字典，一次性画图对比 =====================
    all_spectrums = {
        "MUSIC": P_music,
        "MNM": P_mnm,
        "MVM": P_mvm,
        "MEM": P_mem,
        "Beamspace DFT": P_beam_dft,
        "ESVD": P_esvd,
        "Toeplitz": P_toep,
        "Matrix Decomp": P_decomp,
        "Forward SS": P_ss_f,
        "Backward SS": P_ss_b,
        "Bidirectional SS": P_ss_bi
    }

    music.figure_plot(all_spectrums, true_angles=sim.degsais)

    # ===================== 输出所有估计角度 =====================
    print("=" * 70)
    print(" 所有算法 DOA 估计结果")
    print(f" 真实角度: {sim.degsais}")
    print("-" * 70)
    print(f" MUSIC:         {music.find_peaks(P_music)}")
    print(f" MNM:           {music.find_peaks(P_mnm)}")
    print(f" MVM:           {music.find_peaks(P_mvm)}")
    print(f" MEM:           {music.find_peaks(P_mem)}")
    print(f" Beam DFT:      {music.find_peaks(P_beam_dft)}")
    print(f" ESVD:          {music.find_peaks(P_esvd)}")
    print(f" Toeplitz:      {music.find_peaks(P_toep)}")
    print(f" Matrix Decomp: {music.find_peaks(P_decomp)}")
    print(f" Forward SS:    {music.find_peaks(P_ss_f)}")
    print(f" Backward SS:   {music.find_peaks(P_ss_b)}")
    print(f" Bidirectional SS:{music.find_peaks(P_ss_bi)}")
    print("=" * 70)