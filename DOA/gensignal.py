import numpy as np
from scipy.linalg import sqrtm

class ArraySimulator:
    def __init__(
        self,
        M=8,                # 阵元数
        degsais=[33,36,12], # 信号角度
        sigma2=2,           # 噪声方差（均匀）
        SNR_db=5,           # 信噪比
        N=200,              # 快拍数
        random_seed=42
    ):
        # 固定参数
        self.M = M
        self.degsais = degsais
        self.K = len(degsais)
        self.sigma2 = sigma2
        self.SNR_db = SNR_db
        self.N = N
        self.seed = random_seed
        np.random.seed(random_seed)

    def generate_data(
            self,
            noise_type = "uniform", # 噪声类型
            signal_type = "incoherent" # 信号类型     `
    ):
        if noise_type == "uniform":
            Qv = np.diag([self.sigma2] * self.M)
        elif noise_type == "nonuniform":
            print(f"请输入非均匀噪声的功率数组: {self.M} 个正数")
            sigmanoise = list(map(float, input().split()))
            Qv = np.diag(sigmanoise)
        Qv_sqrt = sqrtm(Qv)
        suminvQv = np.sum(1.0 / np.diag(Qv))
        sigmasq = (self.M / suminvQv) * 10 ** (self.SNR_db / 10)
        m = np.arange(self.M).reshape(-1, 1)
        A = np.exp(1j * np.pi * m @ np.sin(np.deg2rad(self.degsais)).reshape(1, -1))

        if signal_type == "incoherent":
            S = np.sqrt(sigmasq / 2) * (np.random.randn(self.K, self.N) + 1j * np.random.randn(self.K, self.N))
        elif signal_type == "coherent":
            S0 = np.sqrt(sigmasq / 2) * (np.random.randn(1, self.N) + 1j * np.random.randn(1, self.N))
            amps = np.random.uniform(low=0.5, high=1.5, size=self.K)
            amps = amps / np.sqrt(np.mean(amps**2)) 
            S = amps.reshape(-1, 1) * S0
    
        Noise = np.sqrt(1/2) * Qv_sqrt @ (np.random.randn(self.M, self.N) + 1j * np.random.randn(self.M, self.N))
        xmt = A @ S + Noise
        R = (xmt @ xmt.conj().T) / self.N
        return  xmt, R, A, S
     
# 测试你的 ArraySimulator 类
if __name__ == "__main__":
    # 1. 初始化仿真器
    sim = ArraySimulator(
        M=8,                # 阵元
        degsais=[33, 36, 12],# 信号角度
        sigma2=2,           # 噪声功率
        SNR_db=5,           # 信噪比
        N=200,              # 快拍数
        random_seed=42
    )

    # 2. 生成数据：均匀噪声 + 非相干信号
    print("正在生成：均匀噪声 + 非相干信号...")
    xmt, R, A, S = sim.generate_data(
        noise_type="uniform",
        signal_type="incoherent"
    )

    # 3. 输出形状，看是否正确
    print("✅ 接收数据 xmt 形状:", xmt.shape)   # (8, 200)
    print("✅ 协方差矩阵 R 形状:", R.shape)     # (8, 8)
    print("✅ 导向矢量 A 形状:", A.shape)       # (8, 3)
    print("✅ 信号 S 形状:", S.shape)           # (3, 200)
    print("="*50)

    # 4. 测试：非均匀噪声 + 相干信号（功率不同）
    print("正在生成：非均匀噪声 + 相干信号（功率不同）...")
    xmt2, R2, A2, S2 = sim.generate_data(
        noise_type="nonuniform",
        signal_type="coherent"
    )

    print("✅ 接收数据 xmt2 形状:", xmt2.shape)
    print("✅ 协方差矩阵 R2 形状:", R2.shape)
    print("✅ 信号 S2 形状:", S2.shape)
    print("\n🎉 全部运行成功！")