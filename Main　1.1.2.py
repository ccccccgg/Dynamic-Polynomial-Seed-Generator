'''

'此僅支援ASCII輸出'


'''

import os
import sys
import json
import math
import time
import hmac
import random
import struct
import hashlib
import sympy as sp
from decimal import Decimal, getcontext
from sympy.utilities.lambdify import lambdify

# --- 函數：read_config, read_primes, parse_polynomial, generate_prime_combinations, compute_polynomial_values, generate_S1, remove_selected_values ---
# (這些函數保持不變，直接從你提供的代碼複製過來)
def read_config(filename="DPSG system/Config.txt"):
    """讀取 JSON 配置文件"""
    try:
        with open(filename, "r", encoding='utf-8') as file: # 指定 UTF-8
            config = json.load(file)
        if "F1" not in config or "dimension" not in config:
            raise ValueError("配置文件中缺少必要的 'F1' 或 'D'")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {filename} 未找到")
    except json.JSONDecodeError:
        raise ValueError(f"配置文件 {filename} 不是有效的 JSON 格式")
    except Exception as e:
        raise Exception(f"讀取配置文件時發生錯誤: {e}")

def read_primes(filename):
    """讀取 Prime.txt 並返回質數列表"""
    try:
        with open(filename, "r") as file:
            primes = [int(line.strip()) for line in file if line.strip().isdigit()]
        if not primes:
            raise ValueError("質數文件為空或未包含有效的質數")
        return primes
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {filename} 未找到")
    except Exception as e:
        raise Exception(f"讀取質數文件時發生錯誤: {e}")

def parse_polynomial(poly_str):
    """解析多項式字符串並返回 sympy 表達式和變數"""
    try:
        # 允許 Mod 作為函數名
        expr = sp.sympify(poly_str, locals={'Mod': lambda x, y: x % y}) 
        variables = sorted(expr.free_symbols, key=lambda s: str(s))
        # F1 可能沒有變數（常數），F2 會在後面檢查
        # if not variables:
        #     raise ValueError("多項式中未找到有效的變數")
        return expr, variables
    except sp.SympifyError:
        raise ValueError(f"無法解析多項式: {poly_str}")
    except Exception as e:
        raise Exception(f"解析多項式時發生錯誤: {e}")

def generate_prime_combinations(primes, num_variables, D):
    """從質數列表中隨機選擇 num_variables 個質數，生成 D 組"""
    if num_variables > len(primes):
        # 允許使用比質數表更多的變數，但實際只用 len(primes) 個
        print(f"警告: F1 變數數量 ({num_variables}) 大於可用質數數量 ({len(primes)})。將僅使用 {len(primes)} 個質數。", file=sys.stderr)
        num_variables = len(primes)
    if num_variables <= 0: # 如果 F1 是常數，則不需要質數組合
        return [[]] * D # 返回 D 個空列表

    return [random.sample(primes, num_variables) for _ in range(D)]

def compute_polynomial_values(prime_combinations, expr, variables):
    """計算多項式的值"""
    polynomial_values = []
    if not variables: # 如果 F1 是常數
        try:
            const_val = float(expr.evalf())
            # 需要為後續 S1 生成足夠多的值，重複 D*2 次
            # （注意：這會導致 S1 全都一樣，是否期望？）
            return [const_val] * len(prime_combinations) * 2 
        except Exception as e:
            raise ValueError(f"計算常數多項式 F1 時出錯: {e}")
            
    num_expected_vars = len(variables)
    for combination in prime_combinations * 2: # S1 需要 2*D 個基礎值
         # 確保組合長度與變數數量匹配
        if len(combination) != num_expected_vars:
            print(f"警告: 質數組合長度 ({len(combination)}) 與 F1 變數數量 ({num_expected_vars}) 不符，跳過此組合。", file=sys.stderr)
            polynomial_values.append(0.0) # 添加默認值以保持長度
            continue
        subs_dict = dict(zip(variables, combination))
        try:
            value = expr.subs(subs_dict)
            if value.is_number:
                polynomial_values.append(float(value))
            else:
                print(f"警告: F1 計算結果不是數值 ({value})，使用 0.0 代替。", file=sys.stderr)
                polynomial_values.append(0.0)
        except Exception as e:
            print(f"警告: 計算 F1 多項式值時出錯: {e}", file=sys.stderr)
            polynomial_values.append(0.0) # 添加默認值

    return polynomial_values

def generate_S1(A1_values, decimal_places):
    """用 A1(m) 生成初階種子碼，並擷取小數部分後 decimal_places 位"""
    getcontext().prec = decimal_places + 5  # 精度略大於需求，減少額外開銷
    S1 = []
    
    for val in A1_values:
        val = Decimal(val % 1e10)  # 限制數值大小，防止溢出
        d_val = Decimal(val)  # 只轉換一次，減少 Decimal 轉換開銷

        def int_sqrt(val):
            return val.sqrt()  # 直接使用 Decimal.sqrt()

        def int_ln(val):
            """計算對數（使用 float 運算加速）"""
            if val <= 0:
                return Decimal(0)  # 避免無效輸入
            return Decimal(math.log(float(val)))  # 用 math.log 加速計算

        def int_exp(val):
            """計算指數（使用 float 運算加速）"""
            return Decimal(math.exp(float(val)))  # 用 math.exp 加速計算
        
        # **改寫種子碼公式，減少 `Decimal` 轉換次數**
        transformed_val = int_sqrt(d_val) + (d_val % 6 + 1) * int_ln(d_val % 6 + 1) * int_exp(-abs(d_val) % 10)
        
        # **擷取小數部分並補齊 decimal_places 位**
        s = str(transformed_val).split(".")
        if len(s) == 2:  
            S1.append(s[1][:decimal_places].ljust(decimal_places, '0'))
        else:
            S1.append("0" * decimal_places)  # 補 0 避免錯誤
        
    return S1

def remove_selected_values(S1, selected_indices):
    """移除座標軸上的數字並前移"""
    new_S1 = []
    for s in S1:
        s_list = list(s)
        # 確保索引有效且不重複
        valid_indices = sorted(list(set(idx for idx in selected_indices if 0 <= idx < len(s_list))), reverse=True)
        if len(s_list) - len(valid_indices) < 10: # 保留至少 10 位
            print(f"警告: 移除操作可能導致字串 '{s}' 長度小於 10，跳過此字串的移除。", file=sys.stderr)
            new_S1.append(s) # 保留原樣
            continue

        for idx in valid_indices:
            del s_list[idx]
        new_S1.append("".join(s_list))
    return new_S1

# --- 優化版本：限制 internal_state 大小 ---
def generate_ascii_seeds_from_digits(S1, D, F2_expr, F2_func, num_seeds, target_ascii_length):
    total_bits_needed = target_ascii_length * 8
    iterations_needed_for_digits = total_bits_needed

    print(f"為生成長度 {target_ascii_length} 的 ASCII 種子，將使用內部狀態法 (優化版) 從每個 0-9 數字生成 1 個位元。")
    print(f"總共需要生成約 {iterations_needed_for_digits} 個 0-9 數字...")

    # ... (F2 檢查等部分與之前相同) ...
    if not S1 or len(S1) < D:
        raise ValueError(f"S1 內的數據量不足 (需要至少 {D} 個，實際: {len(S1)})")

    F2_variables = sorted(F2_expr.free_symbols, key=lambda s: str(s))
    if len(F2_variables) != D:
        raise ValueError(f"F2 中的變數數量 ({len(F2_variables)}) 與 D ({D}) 不符！")

    ascii_seeds = []
    S1_working_copy = S1[:]

    for seed_idx in range(num_seeds):
        print(f"正在生成第 {seed_idx + 1}/{num_seeds} 組 ASCII 種子...")
        s1_prime_digits = [] # 在每個種子開始時重置

        # --- s1_prime_digits 生成 (與之前 v_custom_state 版本相同) ---
        # (為了簡潔，此處省略，假設 s1_prime_digits 已正確填充)
        if not S1_working_copy or len(S1_working_copy) < D:
            print(f"警告: 第 {seed_idx + 1} 組 - S1 工作副本數據不足 (開始時)。", file=sys.stderr)
            break
        for i in range(iterations_needed_for_digits):
            try:
                if len(S1_working_copy) < D:
                    print(f"警告: 第 {seed_idx + 1} 組, 迭代 {i+1} (digit gen) - S1 工作副本不足 D 個。", file=sys.stderr)
                    s1_prime_digits = []
                    break
                axis_coords_indices = random.sample(range(len(S1_working_copy)), D)
                axis_coords = [S1_working_copy[idx] for idx in axis_coords_indices]
                seed_coords_indices = random.sample(range(len(S1_working_copy)), D)
                seed_coords = [S1_working_copy[idx] for idx in seed_coords_indices]
                if any(not s for s in axis_coords) or any(not s for s in seed_coords): continue # 簡化跳過
                axis_values = []
                valid_iteration = True
                for d_idx in range(D):
                    s_axis = axis_coords[d_idx]
                    if not s_axis or i % len(s_axis) >= len(s_axis) : valid_iteration = False; break
                    axis_values.append(int(s_axis[i % len(s_axis)]) % 10)
                if not valid_iteration: continue
                s_seed = seed_coords[i % D]
                if not s_seed or i % len(s_seed) >= len(s_seed): continue
                F2_int = int(F2_func(*axis_values) % 10)
                seed_digit_val = int(s_seed[i % len(s_seed)]) % 10
                current_s1_digit = (seed_digit_val + F2_int) % 10
                s1_prime_digits.append(current_s1_digit)
            except Exception: continue # 極簡化錯誤處理，專注性能測試
        # --- s1_prime_digits 生成結束 ---

        if not s1_prime_digits or len(s1_prime_digits) < total_bits_needed:
            print(f"資訊: 第 {seed_idx + 1} 組種子未能生成足夠的 0-9 數字 ({len(s1_prime_digits)}/{total_bits_needed})，跳過。", file=sys.stderr)
            continue

        s_binary_bits = []
        internal_state = seed_idx # 讓每個種子的初始狀態略有不同，或者固定為 0
        
        # --- ****** 關鍵優化 ****** ---
        # 選擇一個狀態掩碼，例如32位或64位。64位通常足夠好且高效。
        # Python 的整數運算對於機器字大小的整數是優化的。
        STATE_MASK = 0xFFFFFFFFFFFFFFFF  # 64-bit mask (2**64 - 1)
        # 或者 STATE_MASK = 0xFFFFFFFF      # 32-bit mask (2**32 - 1)

        # 常數也最好在這個範圍內，或者至少不要讓乘法結果輕易超出太多
        MIX_CONST_A = 1103515245 # 一個常用的LCG乘數 (32位)
        MIX_CONST_B = 15287    # LCG加數
        # 如果用64位狀態，可以用64位的常數，例如:
        # MIX_CONST_A_64 = 0x5DEECE66D # Java Random, Python random.Random 用的
        # MIX_CONST_B_64 = 0xB

        for digit_from_s1 in s1_prime_digits:
            # 更新內部狀態，並通過 & STATE_MASK 限制其大小
            # 這裡的公式可以調整，目標是高效且有一定的混合性
            # LCG-like update (線性同餘生成器)
            internal_state = (internal_state * MIX_CONST_A + digit_from_s1 + MIX_CONST_B) & STATE_MASK
            
            # 從狀態中提取位元
            # 方法1: 取最低位 (最快)
            output_bit = (internal_state >> 63) & 1  
            # 方法2: 取最高位 (對於固定大小的狀態，例如取第31位或第63位)
            # output_bit = (internal_state >> 31) & 1 # 如果用32位狀態
            # output_bit = (internal_state >> 63) & 1 # 如果用64位狀態
            # 方法3: 異或折疊 (稍微複雜一點，但可能更好)
            # temp_state = internal_state ^ (internal_state >> 16) # 對於32位狀態
            # temp_state = temp_state ^ (temp_state >> 8)
            # output_bit = temp_state & 1

            s_binary_bits.append(output_bit)
        # --- ****** 優化結束 ****** ---

        # ... (後續位元組轉換和保存不變) ...
        s2_bytes = []
        num_full_bytes = len(s_binary_bits) // 8
        for i in range(num_full_bytes):
            byte_bits = s_binary_bits[i*8 : (i+1)*8]
            byte_string = "".join(map(str, byte_bits))
            try: byte_value = int(byte_string, 2); s2_bytes.append(byte_value)
            except ValueError: pass # 簡化

        try:
            final_ascii_string = "".join([chr(b) for b in s2_bytes])
            ascii_seeds.append(final_ascii_string)
        except ValueError: ascii_seeds.append("") # 簡化
            
    return ascii_seeds

# --- 修改後的 save_output ---
def save_output(seeds, filename):
    """將種子碼保存到文件中 (假設 seeds 是字符串列表)"""
    if not isinstance(seeds, list):
        raise TypeError(f"seeds 必須是 list, 但目前是 {type(seeds)}")

    try:
        # 自動創建輸出目錄
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"已創建輸出目錄: {output_dir}")
            
        content_to_write_str = "".join([str(s) if not isinstance(s, str) else s for s in seeds])
        
        # 將字符串內容按 latin-1 編碼為字節串
        content_bytes = content_to_write_str.encode('latin-1')

        # print(f"DEBUG save_output: 準備寫入的字符串長度: {len(content_to_write_str)}")
        # print(f"DEBUG save_output: 準備寫入的字節串長度: {len(content_bytes)}")

        # 以二進制寫入模式 ("wb") 打開文件
        with open(filename, "wb") as file: # <--- 關鍵是 "wb"
            file.write(content_bytes) # 直接寫入字節串
            
    except OSError as e:
        raise OSError(f"創建輸出目錄或寫入文件時發生 OS 錯誤: {e}")
    except Exception as e:
        raise Exception(f"保存輸出文件 {filename} 時發生錯誤: {e}")


# --- 修改後的 main ---
def main():
    try:
        overall_start_time = time.perf_counter()
        config = read_config()

        save_txt_template = config["save"] # 保留模板
        prime_txt = config["prime"]
        primes = read_primes(prime_txt)

        # ASCII 序列的目標長度來自 config "decimal" 或 "iterations"
        target_ascii_length = config.get("decimal", 100) # 用戶在 config 中設置的是 ASCII 長度
        # 或者使用 config.get("iterations", 100) 如果你換了鍵名

        # 解析 F1
        F1_expr, F1_variables = parse_polynomial(config["F1"])
        num_f1_variables = len(F1_variables)

        # 解析基礎 F2 (用於獲取 D)
        # 注意：這裡不再直接使用 config 中的 F2 字串來生成函數
        # 我們需要 D 來知道要生成多少個 F2 變數
        base_F2_expr, base_F2_variables = parse_polynomial(config["F2"]) # 解析用於獲取 D
        D = config["dimension"] # 直接從 config 讀取維度 D

        # 檢查基礎 F2 的變數數是否與 D 匹配（作為驗證）
        if len(base_F2_variables) != D:
            print(f"警告: Config 文件中原始 F2 的變數數量 ({len(base_F2_variables)}) 與指定的維度 D ({D}) 不符。", file=sys.stderr)
            # 繼續執行，因為我們會動態生成 F2

        num_seeds = config["num_seeds"]

        # --- 為當前維度 D 動態生成 F2 ---
        try:
             # 使用我們之前討論的動態生成函數
            def generate_dynamic_f2_expr(dimension):
                if not 1 <= dimension <= 50:
                    raise ValueError(f"維度必須在 1 到 10 之間")
                base_vars = sp.symbols('a:z')
                current_vars = base_vars[:dimension]
                expr = sp.Integer(1)
                for i, var in enumerate(current_vars):
                    coefficient = 3 if i % 2 == 0 else 3
                    expr += coefficient * var
                return expr % 10, list(current_vars)

            F2_expr_dynamic, F2_variables_dynamic = generate_dynamic_f2_expr(D)
            print(f"使用的 F2 表達式 (維度 {D}): {F2_expr_dynamic}")
            F2_func = lambdify(F2_variables_dynamic, F2_expr_dynamic, "math")
        except Exception as f2_e:
            raise Exception(f"動態生成 F2 函數時出錯: {f2_e}")


        # --- 後續步驟與原版類似，但調用新函數 ---
        prime_combinations = generate_prime_combinations(primes, num_f1_variables, D)
        polynomial_values = compute_polynomial_values(prime_combinations, F1_expr, F1_variables)
        
        # S1 的長度應該是多少？它影響 i % len(s)。
        # 原版 S1 長度由 decimal_places 決定。
        # 這個長度是否需要調整？暫時保持不變。
        s1_decimal_places = config.get("s1_length", target_ascii_length * 8 // (D*2) + 10) # 嘗試估算一個長度? 或保持原樣?
        print(f"生成 S1 使用的 decimal_places: {s1_decimal_places}") # 打印出來看看
        S1 = generate_S1(polynomial_values, s1_decimal_places) 
        if not S1:
            raise Exception("未能生成有效的 S1 列表")

        # --- 調用新的生成函數 ---
        ascii_seeds = generate_ascii_seeds_from_digits(S1, D, F2_expr_dynamic, F2_func, num_seeds, target_ascii_length)

        # --- 保存 (路徑與原版相同) ---
        save_output(ascii_seeds, save_txt_template) # 使用 config 中的原始路徑
        print(f"ASCII 種子碼已生成並保存到: {save_txt_template}")

        end_time = time.perf_counter()
        print(f"總執行時間 {round(end_time - overall_start_time, 5)} 秒")

    except FileNotFoundError as e:
        print(f"錯誤: 文件未找到 - {e}", file=sys.stderr)
    except ValueError as e:
        print(f"錯誤: 配置或數值問題 - {e}", file=sys.stderr)
    except Exception as e:
        print(f"程式執行時發生未預期的錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # 打印詳細錯誤追溯

if __name__ == "__main__":
    main()