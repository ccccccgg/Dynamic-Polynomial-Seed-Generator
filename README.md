# 高強度偽隨機數生成器 (High-Strength PRNG)

<!-- 可選：在這裡放一些徽章，例如自製的 NIST 通過徽章 -->
<!-- ![NIST Tests Passed](link_to_your_badge_image.png) -->

這是一個由高中生獨立設計和實現的高強度偽隨機數生成器 (PRNG)。該項目旨在探索 PRNG 的原理，並挑戰通過國際權威的 NIST SP 800-22 隨機性測試套件。經過多次迭代、調試和優化，該 PRNG 最終成功通過了全部 15 項測試。

This is a high-strength Pseudo-Random Number Generator (PRNG) independently designed and implemented by a high school student. The project aims to explore the principles of PRNGs and to challenge the creation of a generator capable of passing the internationally recognized NIST SP 800-22 randomness test suite. After numerous iterations, debugging sessions, and optimizations, this PRNG successfully passed all 15 tests in the suite.

## 項目描述 (Project Description)

**中文：**

在現代資訊科技中，高品質的隨機數序列是許多應用的基石，包括密碼學、科學模擬和遊戲開發等。本專案的核心目標是從零開始構建一個偽隨機數生成器，它不僅能產生統計上看起來隨機的序列，而且其隨機性要能得到 NIST SP 800-22 測試套件的嚴格驗證。

設計過程涉及多階段的演算法：

1.  **初始熵源生成 (F1 & S1)：** 利用質數組合與多項式計算，結合高精度數學變換來產生初步的數字序列。
2.  **動態混合 (F2)：** 引入一個多維度的混合函數來處理中間序列。
3.  **核心位元生成：** 採用基於內部狀態的機制，通過迭代更新一個受控大小的狀態變數，並從中提取最終的位元流，以增強序列的不可預測性和統計特性。

開發過程中遇到了諸多挑戰，包括演算法的統計偏差、性能瓶頸，以及一個因作業系統換行符轉換導致測試數據失準的意外問題。通過系統性的分析和不懈的努力，這些問題最終都得到了解決。

**English：**

In modern information technology, high-quality random number sequences are fundamental to many applications, including cryptography, scientific simulations, and game development. The core goal of this project was to build a Pseudo-Random Number Generator (PRNG) from scratch that not only produces statistically random-looking sequences but also has its randomness rigorously validated by the NIST SP 800-22 test suite.

The design process involves a multi-stage algorithm:

1.  **Initial Entropy Source Generation (F1 & S1):** Utilizes prime number combinations with polynomial calculations, combined with high-precision mathematical transformations, to generate initial numerical sequences.
2.  **Dynamic Mixing (F2):** Introduces a multi-dimensional mixing function to process intermediate sequences.
3.  **Core Bit Generation:** Employs an internal state-based mechanism. An internal state variable of a controlled size is iteratively updated, and the final bitstream is extracted from this state, aiming to enhance the sequence's unpredictability and statistical properties.

Numerous challenges were encountered during development, including statistical biases in the algorithm, performance bottlenecks, and an unexpected issue where test data была corrupted due to operating system newline conversions. Through systematic analysis and persistent effort, these issues were ultimately resolved.

## 主要特性 (Features)

- 基於自定義多階段演算法設計 (Custom multi-stage algorithm design)
- 引入內部狀態機制以增強序列的不可預測性 (Internal state mechanism to enhance unpredictability)
- **成功通過 NIST SP 800-22 全部 15 項統計測試 (Successfully passed all 15 tests of the NIST SP 800-22 suite)**
- 使用 Python 實現，包含清晰的配置選項 (Implemented in Python with clear configuration options)

## 使用方法 (Usage)

1.  **克隆倉庫 (Clone the repository):**
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git
    cd YourRepositoryName
    ```
2.  **檢查/安裝依賴 (Check/Install dependencies):**
    本項目主要依賴 `sympy`。如果未安裝，請運行：
    ```bash
    pip install sympy
    ```
3.  **配置 (Configuration):**
    - 編輯 `DPSG system/Config.txt` 文件來設置參數，如：
      - `F1`: 多項式表達式 (例如 `"a**9 + ... + 1"`)
      - `F2`: F2 函數的基礎結構 (代碼中會根據 `dimension` 動態生成)
      - `dimension`: F2 函數的維度
      - `num_seeds`: 要生成的種子數量 (通常為 1 進行測試)
      - `decimal`: 目標 ASCII 序列長度 (字符數)
      - `save`: 輸出文件的路徑
      - `prime`: `Prime.txt` 文件的路徑
    - 確保 `DPSG system/Prime.txt` 文件存在且包含一個質數列表 (每行一個質數)。
4.  **運行 (Run):**
    ```bash
    python Main_1.1.1.py
    ```
    生成的 ASCII 序列將保存在 `Config.txt` 中 `save` 字段指定的路徑。

## NIST 測試結果 (NIST Test Results)

本 PRNG 生成的序列已成功通過 NIST SP 800-22 測試套件的所有 15 項測試。
唯有時會出現特定項目無法通過的情況，屬於少數，目前正進行修訂
[NIST Example Output](images/NIST_Example_Output.png)

## 風險與免責聲明 (Risks and Disclaimer)

**中文：**

請注意，這是一個由高中生出於學習和研究目的創建的專案。儘管它已通過 NIST SP 800-22 統計測試套件，但這**並不意味著**它可以直接用於任何對安全性有嚴格要求的生產環境或密碼學應用。

- **學術性質：** 本專案主要用於展示演算法設計和問題解決的過程。
- **未經嚴格密碼分析：** 此 PRNG 未經過專業密碼學家的嚴格分析，可能存在統計測試無法檢測到的潛在弱點。
- **無任何擔保：** 本軟體按“原樣”提供，不作任何明示或暗示的保證，包括但不限於適銷性、特定用途適用性和非侵權性的保證。

任何將此代碼用於實際應用（尤其是有安全需求的場景）的用戶，應自行承擔全部風險，並建議尋求專業安全審計。

**English：**

Please note that this is a project created by a high school student for learning and research purposes. Although it has passed the NIST SP 800-22 statistical test suite, this **does not imply** that it is suitable for direct use in any production environment or cryptographic application with strict security requirements.

- **Academic Nature:** This project primarily serves to demonstrate the process of algorithm design and problem-solving.
- **No Rigorous Cryptanalysis:** This PRNG has not undergone rigorous analysis by professional cryptographers and may possess potential weaknesses undetectable by statistical tests.
- **No Warranty:** The software is provided "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

Users who choose to use this code for practical applications (especially in security-sensitive scenarios) do so at their own risk and are advised to seek professional security audits.

## 許可證 (License)

本項目採用 [MIT License](LICENSE) 授權。
