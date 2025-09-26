import numpy as np
import pandas as pd
import time
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class cgpm:
    def __init__(self, data_path, function_sets=None, input_sets=None, generations=50, population_size=1000):
        # 数据加载
        data = pd.read_csv(data_path)
        self.r = data["r"].values.reshape(-1, 1)
        self.q = 1.0 / self.r
        self.y = data['F'].values
        
        # 自定义函数
        self.smoothing_func = self._create_smoothing_function()
        self.pow_func = self._create_safe_pow_function()
        self.nsum_func = self._create_nsum_function()
        self.aq_func = self._create_aq_function()

        # 设置函数集和输入
        self.function_sets = function_sets if function_sets else [
            ['add', 'sub', 'mul', 'div', self.nsum_func],
            ['add', 'sub', 'mul', 'div', self.nsum_func, self.pow_func],
            ['add', 'sub', 'mul', 'div', self.nsum_func, self.aq_func],
            ['add', 'sub', 'mul', 'div', self.nsum_func, self.aq_func, self.pow_func]
        ]

        self.input_sets = input_sets if input_sets else {
            'r': self.r,
            'q': self.q,
            'r+q': np.hstack([self.r, self.q])
        }

        self.generations = generations
        self.population_size = population_size

    def _create_smo
