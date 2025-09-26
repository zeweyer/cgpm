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

    def _create_smoothing_function(self):
        def smoothing_func(r, rin=3.0, rout=5.0):
            r_clipped = np.clip(r, rin, rout)
            t = (r_clipped - rin) / (rout - rin)
            return 2 * t**3 - 3 * t**2 + 1
        return make_function(function=smoothing_func, name='smooth', arity=1)

    def _create_safe_pow_function(self):
        def _safe_pow(x, y):
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                base = np.abs(x)
                exp = np.clip(y, -5, 5)
                res = np.power(base, exp)
                res = np.clip(res, -1e6, 1e6)  # 防止爆炸
                valid_mask = np.isfinite(res)
                mean_val = np.nanmean(res[valid_mask]) if np.any(valid_mask) else 1.0
                res[~valid_mask] = mean_val
            return res
        return make_function(function=_safe_pow, name='pow', arity=2)

    def _create_nsum_function(self):
        def _nsum(x, y):
            return x + y
        return make_function(function=_nsum, name='nsum', arity=2)

    def _create_aq_function(self):
        def _aq(x):
            return np.sqrt(np.power(x, 2) + 1)
        return make_function(function=_aq, name='aq', arity=1)

    def __call__(self):
        # 最终调用，开始训练过程
        self.train()

    def train(self):
        results = []
        model_id = 0

        for input_name, X in self.input_sets.items():
            X_train, X_val, y_train, y_val = train_test_split(X, self.y, test_size=0.2, random_state=0)

            for fset in self.function_sets:
                model_id += 1
                start = time.time()

                try:
                    est_gp = SymbolicRegressor(
                        population_size=self.population_size,
                        generations=self.generations,
                        tournament_size=20,
                        stopping_criteria=1e-5,
                        const_range=(-10, 10),
                        init_depth=(2, 6),
                        init_method='half and half',
                        function_set=fset,
                        parsimony_coefficient=0.01,
                        max_samples=0.9,
                        verbose=0,
                        random_state=42,
                        n_jobs=4
                    )
                    est_gp.fit(X_train, y_train)
                    runtime = time.time() - start

                    # 预测 + 过滤非法值
                    y_train_pred = est_gp.predict(X_train)
                    y_val_pred = est_gp.predict(X_val)

                    train_mask = np.isfinite(y_train_pred)
                    val_mask = np.isfinite(y_val_pred)

                    if not np.all(train_mask):
                        print(f"⚠️ 模型 {model_id} 训练集预测有 {np.sum(~train_mask)} 个无效值，已过滤")
                    if not np.all(val_mask):
                        print(f"⚠️ 模型 {model_id} 验证集预测有 {np.sum(~val_mask)} 个无效值，已过滤")

                    y_train_clean = y_train[train_mask]
                    y_train_pred_clean = y_train_pred[train_mask]
                    y_val_clean = y_val[val_mask]
                    y_val_pred_clean = y_val_pred[val_mask]

                    train_mae = mean_absolute_error(y_train_clean, y_train_pred_clean)
                    test_mae = mean_absolute_error(y_val_clean, y_val_pred_clean)
                    fitness = mean_squared_error(y_val_clean, y_val_pred_clean)

                    program = est_gp._program
                    complexity = program.length_
                    computational_cost = str(program).count('nsum')

                    joblib.dump(est_gp, f"gp_model_{model_id:02d}.pkl")

                    results.append([
                        model_id,
                        '+'.join([f.name if hasattr(f, "name") else f for f in fset]),
                        input_name,
                        train_mae,
                        test_mae,
                        complexity,
                        runtime,
                        complexity,
                        computational_cost,
                        fitness
                    ])

                except Exception as e:
                    print(f"❌ 模型 {model_id} 训练失败，错误：{e}")
                    results.append([
                        model_id,
                        '+'.join([f.name if hasattr(f, "name") else f for f in fset]),
                        input_name,
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    ])

        # 保存结果
        columns = ["ID", "primitive set", "inputs", "MAE train", "MAE test", "length", "runtime", "complexity", "costs", "fitness"]
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv("gp_comparison_summary.csv", index=False)
        print("\n✅ 回归完成，已保存模型结果至 gp_comparison_summary.csv")
        print(results_df)

# 用法示例
if __name__ == '__main__':
    model = cgpm(data_path="analysis/new_force_pairs.csv")
    model()  # 通过调用 cgpm 实例来启动训练过程
