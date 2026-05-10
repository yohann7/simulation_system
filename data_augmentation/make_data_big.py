import numpy as np
import pandas as pd
from pathlib import Path

def calculate_data_attributes(origenal_data):
    data_attributes = pd.DataFrame(
        [origenal_data.mean(axis=0), origenal_data.var(axis=0)],
        index=["mean", "variance"],
    )
    return data_attributes

def Jittering(origenal_data, data_attributes, n):
    if n <= 0:
        return origenal_data.iloc[0:0].copy()

    columns = origenal_data.columns.tolist()
    original_row_set = {
        tuple(row)
        for row in origenal_data[columns].itertuples(index=False, name=None)
    }

    element_headers = {
        "C_ELE",
        "SI_ELE",
        "MN_ELE",
        "P_ELE",
        "S_ELE",
        "CR_ELE",
        "NI_ELE",
        "CU_ELE",
    }
    normalized_cols = {col: str(col).strip().upper() for col in columns}
    element_cols = [col for col in columns if normalized_cols[col] in element_headers]
    temp_cols = [
        col
        for col in columns
        if str(col).strip().endswith("(0)") or str(col).strip().endswith("(1)")
    ]

    if len(element_cols) == 0:
        raise ValueError("未识别到元素列，请检查是否包含 C_ELE~CU_ELE 表头。")
    if len(temp_cols) == 0:
        raise ValueError("未识别到温度列，请检查温度列名是否以 (0) 或 (1) 结尾。")

    variances = data_attributes.loc["variance", columns].clip(lower=0.0)
    ts_col = next((col for col in columns if str(col).strip().upper() == "TS"), None)

    element_std = np.sqrt(variances[element_cols].to_numpy(dtype=float)) if element_cols else np.array([])
    temp_std = np.sqrt(variances[temp_cols].to_numpy(dtype=float)) if temp_cols else np.array([])
    ts_std = float(np.sqrt(variances[ts_col])) if ts_col is not None else 0.0

    def _zero_mean_and_bound(noise_mat, limit):
        if noise_mat.size == 0:
            return noise_mat
        # 先做列方向去均值，保证每一列在n条虚拟数据中的平均改变量为0。
        centered = noise_mat - noise_mat.mean(axis=0, keepdims=True)
        if limit is None or limit <= 0:
            return centered
        max_abs = np.max(np.abs(centered), axis=0, keepdims=True)
        scale = np.where(max_abs > limit, limit / max_abs, 1.0)
        return centered * scale

    def element_Jittering(elemnts_vec, k):
        if elemnts_vec.size == 0:
            return np.empty((n, 0))
        noise = np.random.normal(loc=0.0, scale=element_std * k, size=(n, elemnts_vec.size))
        noise = _zero_mean_and_bound(noise, limit=0.05)
        new_elemnts_vec = elemnts_vec[np.newaxis, :] + noise
        return new_elemnts_vec

    def T_Jittering(T_vec, k):
        if T_vec.size == 0:
            return np.empty((n, 0))
        noise = np.random.normal(loc=0.0, scale=temp_std * k, size=(n, T_vec.size))
        noise = _zero_mean_and_bound(noise, limit=1.0)
        new_T_vec = T_vec[np.newaxis, :] + noise
        return new_T_vec

    def Ts_Jittering(TS, k=0):
        if ts_col is None:
            return np.array([])
        if k == 0:
            return np.full(n, TS, dtype=float)
        delta_TS = np.random.normal(loc=0.0, scale=ts_std * k, size=n)
        delta_TS = _zero_mean_and_bound(delta_TS.reshape(n, 1), limit=3.0).reshape(n)
        new_TS = TS + delta_TS
        return new_TS

    def _row_key(row):
        return tuple(row[col] for col in columns)

    def _force_row_different(new_row):
        if _row_key(new_row) not in original_row_set:
            return new_row

        jitter_cols = element_cols + temp_cols + ([ts_col] if ts_col is not None else [])
        if not jitter_cols:
            raise ValueError("没有可用于扰动的列，无法保证生成数据与原数据不同。")

        for col in jitter_cols:
            base_value = float(new_row[col])
            step = np.nextafter(base_value, np.inf) - base_value
            if step == 0:
                step = 1e-12

            candidate = new_row.copy()
            candidate.loc[col] = base_value + step
            if _row_key(candidate) not in original_row_set:
                return candidate

            candidate = new_row.copy()
            candidate.loc[col] = base_value - step
            if _row_key(candidate) not in original_row_set:
                return candidate

        raise ValueError("生成的数据与原数据完全相同，请检查扰动幅度或原始数据是否过于离散。")

    k_element = 0.08
    k_T = 0.05
    k_TS = 0

    augmented_rows = []
    for _, row in origenal_data.iterrows():
        elemnts_vec = row[element_cols].to_numpy(dtype=float) if element_cols else np.array([])
        T_vec = row[temp_cols].to_numpy(dtype=float) if temp_cols else np.array([])

        new_elemnts_mat = element_Jittering(elemnts_vec, k_element)
        new_T_mat = T_Jittering(T_vec, k_T)
        new_TS_vec = Ts_Jittering(row[ts_col], k_TS) if ts_col is not None else np.array([])

        generated_rows = []
        for i in range(n):
            new_row = row.copy()
            if element_cols:
                new_row.loc[element_cols] = new_elemnts_mat[i]
            if temp_cols:
                new_row.loc[temp_cols] = new_T_mat[i]
            if ts_col is not None:
                new_row.loc[ts_col] = new_TS_vec[i]
            generated_rows.append(_force_row_different(new_row))

        augmented_rows.append(pd.DataFrame(generated_rows, columns=columns))

    jittered_data = pd.concat(augmented_rows, ignore_index=True)
    return jittered_data

def all_in_one(augmented_data, origenal_data):
    if not isinstance(augmented_data, pd.DataFrame) or not isinstance(origenal_data, pd.DataFrame):
        raise TypeError("all_in_one 的输入必须是两个 pandas.DataFrame。")
    all_data = pd.concat([origenal_data, augmented_data], ignore_index=True, sort=False)
    return all_data

if __name__ == "__main__":
    origenal_data_path = Path(__file__).parents[1] /"sim_T"/"output_data"/ "process_data_new.csv"
    origenal_data = pd.read_csv(origenal_data_path)
    data_attributes = calculate_data_attributes(origenal_data)
    augmented_data = Jittering(origenal_data, data_attributes, n=10)
    print(augmented_data.shape)
    augmented_data.to_csv(Path(__file__).parent / "output_data" / "augmented_process_data.csv", index=False)
    # augmented_data.to_csv(r"math_sim\82A\v6（yang数据）\data_augmentation\augmented_process_data.csv", index=False)
    all_data = all_in_one(augmented_data, origenal_data)
    print(all_data.shape)
    # all_data.to_csv(r"math_sim\82A\v6（yang数据）\data_augmentation\all_process_data.csv", index=False)
    all_data.to_csv(Path(__file__).parent / "output_data" / "all_process_data.csv", index=False)