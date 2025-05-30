import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional
import pandas as pd

class BaseDriftDetector:
    """漂移检测器的基类"""
    def __init__(self):
        self.warning_detected = False
        self.drift_detected = False
        self.history = {
            'error_rate': [],
            'warning_level': [],
            'drift_level': [],
            'detection_level': []
        }
    
    def reset(self):
        """重置检测器状态"""
        raise NotImplementedError
    
    def update(self, error: Union[bool, int, float]):
        """更新检测器状态"""
        raise NotImplementedError
    
    def plot_history(self, save_path: Optional[str] = None):
        """绘制检测历史"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['error_rate'], label='Error Rate', color='blue')
        plt.plot(self.history['warning_level'], label='Warning Level', color='orange', linestyle='--')
        plt.plot(self.history['drift_level'], label='Drift Level', color='red', linestyle='--')
        plt.plot(self.history['detection_level'], label='Detection Level', color='green')
        
        # 标记警告和漂移点
        warning_points = np.where(np.array(self.history['detection_level']) > self.warning_level)[0]
        drift_points = np.where(np.array(self.history['detection_level']) > self.drift_level)[0]
        
        plt.scatter(warning_points, np.array(self.history['detection_level'])[warning_points], 
                   color='orange', label='Warning Detected', marker='^')
        plt.scatter(drift_points, np.array(self.history['detection_level'])[drift_points], 
                   color='red', label='Drift Detected', marker='*')
        
        plt.title('Drift Detection History')
        plt.xlabel('Sample Index')
        plt.ylabel('Error Rate / Level')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class DDM(BaseDriftDetector):
    """
    Drift Detection Method (DDM)
    
    Reference:
    Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
    Learning with drift detection. In Brazilian symposium on artificial intelligence (pp. 286-295).
    """
    
    def __init__(self, min_num_instances: int = 30, warning_level: float = 2.0, drift_level: float = 3.0):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()
    
    def reset(self):
        """重置检测器"""
        super().reset()
        self.n_i = 1
        self.p_i = 1
        self.s_i = 0
        self.psi = 0
        
        self.n_min = None
        self.p_min = None
        self.s_min = None
        self.psmin = None
    
    def update(self, error: Union[bool, int, float]):
        """
        更新检测器状态
        
        Args:
            error: 0表示正确预测，1表示错误预测
        """
        if isinstance(error, bool):
            error = 1 if error else 0
        
        # 增加实例计数
        self.n_i += 1
        
        # 更新错误率
        self.p_i = self.p_i + (error - self.p_i) / self.n_i
        self.s_i = np.sqrt(self.p_i * (1 - self.p_i) / self.n_i)
        
        self.warning_detected = False
        self.drift_detected = False
        
        if self.n_i < self.min_num_instances:
            return
        
        if self.n_min is None or self.p_i + self.s_i < self.p_min + self.s_min:
            self.n_min = self.n_i
            self.p_min = self.p_i
            self.s_min = self.s_i
            self.psmin = self.p_min + self.s_min
        
        # 检测漂移
        current_level = (self.p_i + self.s_i - self.psmin) / self.s_min
        
        # 更新历史记录
        self.history['error_rate'].append(self.p_i)
        self.history['warning_level'].append(self.warning_level)
        self.history['drift_level'].append(self.drift_level)
        self.history['detection_level'].append(current_level)
        
        if current_level > self.drift_level:
            self.drift_detected = True
            self.reset()
        elif current_level > self.warning_level:
            self.warning_detected = True

class EDDM(BaseDriftDetector):
    """
    Early Drift Detection Method (EDDM)
    
    Reference:
    Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavaldà, R., & Morales-Bueno, R. (2006).
    Early drift detection method. In Fourth international workshop on knowledge discovery from data streams (pp. 77-86).
    """
    
    def __init__(self, min_num_instances: int = 30, warning_level: float = 0.95, drift_level: float = 0.9):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()
    
    def reset(self):
        """重置检测器"""
        super().reset()
        self.n_i = 0
        self.p_i = 0
        self.s_i = 0
        self.p2_i = 0
        
        self.n_max = None
        self.p_max = None
        self.s_max = None
        self.p2_max = None
    
    def update(self, error: Union[bool, int, float]):
        """
        更新检测器状态
        
        Args:
            error: 0表示正确预测，1表示错误预测
        """
        if isinstance(error, bool):
            error = 1 if error else 0
        
        # 增加实例计数
        self.n_i += 1
        
        # 更新错误率
        self.p_i = self.p_i + (error - self.p_i) / self.n_i
        self.p2_i = self.p2_i + (error * error - self.p2_i) / self.n_i
        self.s_i = np.sqrt(self.p2_i - self.p_i * self.p_i)
        
        self.warning_detected = False
        self.drift_detected = False
        
        if self.n_i < self.min_num_instances:
            return
        
        if self.n_max is None or self.p_i + 2 * self.s_i > self.p_max + 2 * self.s_max:
            self.n_max = self.n_i
            self.p_max = self.p_i
            self.s_max = self.s_i
            self.p2_max = self.p2_i
        
        # 检测漂移
        current_level = (self.p_i + 2 * self.s_i) / (self.p_max + 2 * self.s_max)
        
        # 更新历史记录
        self.history['error_rate'].append(self.p_i)
        self.history['warning_level'].append(self.warning_level)
        self.history['drift_level'].append(self.drift_level)
        self.history['detection_level'].append(current_level)
        
        if current_level < self.drift_level:
            self.drift_detected = True
            self.reset()
        elif current_level < self.warning_level:
            self.warning_detected = True

def detect_drift_in_sequence(sequence: np.ndarray, detector: BaseDriftDetector, 
                           window_size: int = 100) -> dict:
    """
    在序列数据中检测漂移
    
    Args:
        sequence: 输入序列
        detector: 漂移检测器实例
        window_size: 滑动窗口大小
    
    Returns:
        包含检测结果的字典
    """
    results = {
        'drift_points': [],
        'warning_points': [],
        'error_rates': [],
        'detection_levels': []
    }
    
    for i in range(0, len(sequence), window_size):
        window = sequence[i:i+window_size]
        error_rate = np.mean(window)
        
        detector.update(error_rate)
        results['error_rates'].append(error_rate)
        results['detection_levels'].append(detector.history['detection_level'][-1])
        
        if detector.drift_detected:
            results['drift_points'].append(i)
        if detector.warning_detected:
            results['warning_points'].append(i)
    
    return results