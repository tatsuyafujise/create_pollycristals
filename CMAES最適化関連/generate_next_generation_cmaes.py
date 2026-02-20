import numpy as np
import os
import csv
import pickle

# CMA-ES状態管理用のグローバル変数
_cmaes_instance = None

def read_individual_from_file(file_path):
    """個体ファイルを読み込んで、フラット化された配列として返す"""
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                x, y, z = map(float, line.strip().split())
                coords.extend([x, y, z])
    return np.array(coords)

def write_individual_to_file(individual, file_path, num_points=15):
    """フラット化された個体を座標ファイル形式で保存"""
    coords = individual.reshape(num_points, 3)
    with open(file_path, 'w') as f:
        for i in range(num_points):
            x, y, z = coords[i]
            # 境界制約を適用
            x = np.clip(x, 0, 24.0)
            y = np.clip(y, 0, 9.0)
            z = np.clip(z, 0, 8.0)
            f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")

def read_fitness_from_csv(csv_file, generation_num):
    """CSVファイルから指定された世代の適応度を読み込む"""
    if not os.path.exists(csv_file):
        return None
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # 世代を探す
    target_gen_name = f"第{generation_num}世代"
    for row in rows:
        if row and row[0] == target_gen_name:
            # 適応度値を数値に変換（NaNは除外）
            fitness_values = []
            for val in row[1:]:
                try:
                    fitness_values.append(float(val))
                except (ValueError, TypeError):
                    fitness_values.append(float('inf'))  # 無効な値は最悪値として扱う
            return np.array(fitness_values)
    
    return None

class CMAES:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) の実装"""
    
    def __init__(self, dimension, population_size=None, sigma0=0.5):
        self.dimension = dimension
        self.lambda_ = population_size if population_size else 4 + int(3 * np.log(dimension))
        self.mu = self.lambda_ // 2
        self.sigma = sigma0
        
        # 重み計算
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        
        # 戦略パラメータ
        self.cc = (4 + self.mu_eff / dimension) / (dimension + 4 + 2 * self.mu_eff / dimension)
        self.cs = (self.mu_eff + 2) / (dimension + self.mu_eff + 5)
        self.c1 = 2 / ((dimension + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((dimension + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dimension + 1)) - 1) + self.cs
        
        # 動的変数
        self.mean = np.zeros(dimension)
        self.pc = np.zeros(dimension)
        self.ps = np.zeros(dimension)
        self.C = np.eye(dimension)
        self.invsqrtC = np.eye(dimension)
        self.sqrtC = np.eye(dimension)
        self.eigeneval = 0
        self.counteval = 0
        
        # 期待値
        self.chiN = np.sqrt(dimension) * (1 - 1/(4*dimension) + 1/(21*dimension**2))
        
    def set_initial_mean(self, mean):
        """初期平均値を設定"""
        self.mean = np.array(mean)
        
    def ask(self):
        """新しい候補解を生成"""
        # 固有値分解（定期的に実行）
        if self.counteval - self.eigeneval > self.lambda_ / (self.c1 + self.cmu) / self.dimension / 10:
            self.eigeneval = self.counteval
            self.C = (self.C + self.C.T) / 2  # 対称性を強制
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-14)  # 数値安定性
            self.sqrtC = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            self.invsqrtC = eigenvectors @ np.diag(1/np.sqrt(eigenvalues)) @ eigenvectors.T
        
        # 候補解生成
        solutions = []
        for _ in range(self.lambda_):
            z = np.random.standard_normal(self.dimension)
            y = self.sqrtC @ z
            x = self.mean + self.sigma * y
            solutions.append(x)
        
        self.solutions = np.array(solutions)
        return self.solutions
    
    def tell(self, fitness_values):
        """適応度に基づいて分布を更新"""
        # ソート（最小化問題）
        indices = np.argsort(fitness_values)
        self.solutions = self.solutions[indices]
        fitness_values = fitness_values[indices]
        
        # 選択された個体
        selected = self.solutions[:self.mu]
        
        # 平均更新
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * selected, axis=0)

        # ステップサイズ制御
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.invsqrtC @ (self.mean - old_mean) / self.sigma)

        # 共分散行列更新
        # 分母が0にならないよう小さな値で安定化
        denom_sq = 1 - (1 - self.cs)**(2 * self.counteval / max(1, self.lambda_))
        denom = np.sqrt(max(denom_sq, 1e-16))
        hsig = (np.linalg.norm(self.ps) / denom < 1.4 + 2 / (self.dimension + 1))

        self.pc = (1 - self.cc) * self.pc + (hsig.astype(float) * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - old_mean) / self.sigma)

        # 共分散行列の更新
        artmp = (selected - old_mean) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * np.sum(self.weights[:, np.newaxis, np.newaxis] * np.array([np.outer(art, art) for art in artmp]), axis=0))

        # ステップサイズ更新
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        # σの下限を設定（探索が停滞しないように）
        min_sigma = 0.01  # 最小σ
        if self.sigma < min_sigma:
            self.sigma = min_sigma
            print(f"Warning: σが下限{min_sigma}に達しました。探索を維持します。")

        self.counteval += self.lambda_

        return self.mean, self.sigma, fitness_values[0]

def save_cmaes_state(cmaes, state_file):
    """CMA-ES状態をファイルに保存"""
    try:
        state = {
            'mean': cmaes.mean,
            'sigma': cmaes.sigma,
            'C': cmaes.C,
            'pc': cmaes.pc,
            'ps': cmaes.ps,
            'counteval': cmaes.counteval,
            'dimension': cmaes.dimension
        }
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"CMA-ES状態を保存: {state_file}")
    except Exception as e:
        print(f"状態保存に失敗: {e}")

def load_cmaes_state(state_file, dimension, population_size=20):
    """CMA-ES状態をファイルから読み込み"""
    try:
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            cmaes = CMAES(dimension, population_size, sigma0=state['sigma'])
            cmaes.mean = state['mean']
            cmaes.sigma = state['sigma']
            cmaes.C = state['C']
            cmaes.pc = state['pc']
            cmaes.ps = state['ps']
            cmaes.counteval = state['counteval']
            print(f"CMA-ES状態を復元: {state_file} (世代: {state['counteval']//population_size})")
            return cmaes
        else:
            print(f"状態ファイルが見つかりません: {state_file}")
            return None
    except Exception as e:
        print(f"状態読み込みに失敗: {e}")
        return None

def generate_next_generation_cmaes(population_folder, fitness_filename, output_folder, generation_num):
    """CMAESを使用して次世代を生成"""
    print(f"CMAES による第{generation_num+1}世代の生成を開始...")
    
    # 現在の個体群を読み込み
    individuals = []
    for i in range(1, 21):  # 20個体
        file_path = os.path.join(population_folder, f"individual_{i:03d}.txt")
        if os.path.exists(file_path):
            individual = read_individual_from_file(file_path)
            individuals.append(individual)
        else:
            print(f"警告: {file_path} が見つかりません")
    
    if not individuals:
        print("エラー: 読み込める個体がありません")
        return
    
    individuals = np.array(individuals)
    print(f"読み込んだ個体数: {len(individuals)}")
    
    # 適応度を読み込み
    fitness_values = read_fitness_from_csv(fitness_filename, generation_num)
    if fitness_values is None:
        print(f"エラー: 第{generation_num}世代の適応度データが見つかりません")
        return
    
    # NaN値の処理: NaNを大きな値（ペナルティ）で置換
    original_fitness = fitness_values.copy()
    valid_mask = ~np.isnan(fitness_values)
    
    if np.sum(valid_mask) == 0:
        print("エラー: 全ての適応度がNaNです")
        return
    
    # NaNを有効な値の最大値 + ペナルティで置換
    valid_fitness = fitness_values[valid_mask]
    max_valid_fitness = np.max(valid_fitness)
    penalty_value = max_valid_fitness + 10.0
    fitness_values[~valid_mask] = penalty_value
    
    print(f"適応度データ: 有効個体={np.sum(valid_mask)}/20, 平均={np.mean(valid_fitness):.4f}, 最小={np.min(valid_fitness):.4f}")
    print(f"NaN個体数: {np.sum(~valid_mask)}, ペナルティ値: {penalty_value:.4f}")
    
    # CMA-ES状態ファイルのパス
    state_file = os.path.join(os.path.dirname(output_folder), "cmaes_state.pkl")
    
    # 前世代のCMA-ES状態を読み込み（存在する場合）
    dimension = len(individuals[0])  # 15 * 3 = 45次元
    cmaes = load_cmaes_state(state_file, dimension, population_size=20)
    
    # 状態の異常を検出してリセット
    needs_reset = False
    if cmaes is not None:
        # 異常検出条件
        if cmaes.sigma > 1000.0 or cmaes.sigma < 1e-10:
            print(f"Warning: σが異常な値({cmaes.sigma:.2e})です。CMA-ESをリセットします。")
            needs_reset = True
        elif np.sum(cmaes.mean != 0) == 0:
            print(f"Warning: 平均が全て0です。CMA-ESをリセットします。")
            needs_reset = True
        elif np.any(np.isnan(cmaes.mean)) or np.any(np.isinf(cmaes.mean)):
            print(f"Warning: 平均にNaNまたはInfが含まれています。CMA-ESをリセットします。")
            needs_reset = True
        elif np.any(np.isnan(cmaes.C)) or np.any(np.isinf(cmaes.C)):
            print(f"Warning: 共分散行列にNaNまたはInfが含まれています。CMA-ESをリセットします。")
            needs_reset = True
    
    if needs_reset:
        cmaes = None
        # 古い状態ファイルを削除
        if os.path.exists(state_file):
            os.remove(state_file)
            print(f"異常な状態ファイルを削除しました: {state_file}")
    
    if cmaes is None:
        # 新規作成（第1-2世代または状態ファイルがない場合）
        # cmaes = CMAES(dimension, population_size=20, sigma0=0.05)  # 3000世代用に初期σを小さく調整
        # # 現在の最良個体を初期平均に設定
        # best_idx = np.argmin(fitness_values)
        # cmaes.set_initial_mean(individuals[best_idx])
        # 初期σをやや大きめにして探索を広げる（必要なら調整）
        cmaes = CMAES(dimension, population_size=20, sigma0=0.20)
        
        # 最良個体を使うが、全て0の場合はランダムに初期化
        best_idx = np.argmin(fitness_values)
        initial_mean = individuals[best_idx].copy()
        
        # 全て0の場合はランダムに初期化
        if np.all(initial_mean == 0):
            print("Warning: 最良個体が全て0のため、ランダムに初期化します")
            lower = np.array([0.0, 0.0, 0.0] * (dimension // 3))
            upper = np.array([24.0, 9.0, 8.0] * (dimension // 3))
            initial_mean = np.random.uniform(lower + 1.0, upper - 1.0)
        
        cmaes.set_initial_mean(initial_mean)
        
        # 平均が境界に張り付いているとサンプリング後すべてが境界に張り付くので軽く内側へノッジ
        # 各点の境界（x,y,z を繰り返し）
        lower = np.array([0.0, 0.0, 0.0] * (dimension // 3))
        upper = np.array([24.0, 9.0, 8.0] * (dimension // 3))
        eps = max(1.0, 1.0 * cmaes.sigma)  # ノッジ量を増加
        m = cmaes.mean
        m = np.where(m < lower + eps, lower + eps, m)
        m = np.where(m > upper - eps, upper - eps, m)
        cmaes.mean = m
        print(f"新規CMA-ESインスタンス作成 (σ={cmaes.sigma:.4f}, 初期平均: min={m.min():.3f}, max={m.max():.3f})")
    else:
        print(f"前世代のCMA-ES状態を復元 (σ={cmaes.sigma:.4f})")
    
    # 平均値が境界に張り付いている場合、内側へ押し戻す
    lower = np.array([0.0, 0.0, 0.0] * (dimension // 3))
    upper = np.array([24.0, 9.0, 8.0] * (dimension // 3))
    eps = max(0.5, 2.0 * cmaes.sigma)  # 押し戻し量を増加（σの2倍以上）
    m = cmaes.mean
    stuck_at_boundary = False
    
    # 境界に張り付いている次元をチェック
    for i in range(len(m)):
        if m[i] <= lower[i] + 0.01:  # ほぼ下限
            m[i] = lower[i] + eps
            stuck_at_boundary = True
        elif m[i] >= upper[i] - 0.01:  # ほぼ上限
            m[i] = upper[i] - eps
            stuck_at_boundary = True
    
    if stuck_at_boundary:
        cmaes.mean = m
        print(f"Warning: 平均が境界に張り付いていたため、内側へ修正しました（eps={eps:.3f}）")
    
    print(f"最良個体のインデックス: {np.argmin(fitness_values)+1}, 適応度: {np.min(fitness_values):.4f}")
    
    # **重要**: 現在の個体群をCMAESに設定してから学習
    cmaes.solutions = individuals
    
    # 適応度の多様性をチェック
    fitness_std = np.std(fitness_values[valid_mask])
    fitness_range = np.max(fitness_values[valid_mask]) - np.min(fitness_values[valid_mask])
    print(f"適応度の多様性: 標準偏差={fitness_std:.6f}, 範囲={fitness_range:.6f}")
    
    if fitness_std < 1e-6 or fitness_range < 1e-6:
        print(f"Warning: 適応度がほぼ同じです。ノイズを追加して多様性を確保します。")
        # 小さなノイズを追加
        noise = np.random.uniform(-0.1, 0.1, size=len(fitness_values))
        fitness_values = fitness_values + noise
    
    # **重要**: 現在の個体群の適応度を使ってCMA-ESを更新
    mean_before = cmaes.mean.copy()
    sigma_before = cmaes.sigma
    # tell() は内部状態を更新する（戻り値に依存しない）
    cmaes.tell(fitness_values)
    updated_mean = cmaes.mean
    updated_sigma = cmaes.sigma
    print(f"CMA-ES更新: σ {sigma_before:.4f} → {updated_sigma:.4f}")
    print(f"平均移動距離: {np.linalg.norm(updated_mean - mean_before):.4f}")
    
    # σの範囲チェックと修正
    if updated_sigma < 0.005:
        print(f"Warning: σが極端に小さい({updated_sigma:.6f})ため、0.05にリセットします")
        cmaes.sigma = 0.05
    elif updated_sigma > 10.0:
        print(f"Warning: σが極端に大きい({updated_sigma:.2e})ため、1.0にリセットします")
        cmaes.sigma = 1.0
    elif np.isnan(updated_sigma) or np.isinf(updated_sigma):
        print(f"Warning: σが異常値({updated_sigma})ため、0.2にリセットします")
        cmaes.sigma = 0.2
    # 詳細診断ログ（可視化・原因特定のため）
    try:
        eigvals = np.linalg.eigvalsh(cmaes.C)
        print(f"診断: trace(C)={np.trace(cmaes.C):.6g}, eig_min={eigvals.min():.6g}, eig_max={eigvals.max():.6g}")
        print(f"       mean[min,max]={updated_mean.min():.6g},{updated_mean.max():.6g}, norm(ps)={np.linalg.norm(cmaes.ps):.6g}")
    except Exception:
        pass
    
    # CMA-ES状態を保存
    save_cmaes_state(cmaes, state_file)
    
    # 新しい候補解を生成（再サンプリング方式で境界を扱う）
    new_solutions = []
    lower = np.array([0.0, 0.0, 0.0] * (dimension // 3))
    upper = np.array([24.0, 9.0, 8.0] * (dimension // 3))
    max_resample = 8  # 再サンプリング回数（調整可）
    total_reflections = 0
    total_resamples = 0
    inside_count = 0

    for _ in range(cmaes.lambda_):
        accepted = None
        tries = 0
        while tries < max_resample:
            tries += 1
            z = np.random.standard_normal(cmaes.dimension)
            try:
                y = cmaes.sqrtC @ z
            except Exception:
                y = z
            x = cmaes.mean + cmaes.sigma * y
            # 範囲内なら受け入れ
            if np.all(x >= lower) and np.all(x <= upper):
                accepted = x
                inside_count += 1
                total_resamples += (tries - 1)
                break
        if accepted is None:
            # 再サンプリング失敗時: 反射して最終クリップ（フォールバック）
            refl = x.copy()
            for i in range(len(refl)):
                if refl[i] < lower[i]:
                    refl[i] = lower[i] + (lower[i] - refl[i])
                if refl[i] > upper[i]:
                    refl[i] = upper[i] - (refl[i] - upper[i])
                # 最後に安全に収める
                refl[i] = np.clip(refl[i], lower[i], upper[i])
            accepted = refl
            total_reflections += 1
        new_solutions.append(accepted)

    new_solutions = np.array(new_solutions)
    # ログ: 境界挙動の統計
    print(f"生成統計: inside_count={inside_count}/{cmaes.lambda_}, total_resamples={total_resamples}, total_reflections={total_reflections}")
    # 追加診断: 何個が上下限に正確に貼り付いているか（要注意）
    at_lower = np.sum(new_solutions == lower, axis=None)
    at_upper = np.sum(new_solutions == upper, axis=None)
    print(f"貼り付き検出: lower_pts={int(at_lower)}, upper_pts={int(at_upper)}")
    # 出力フォルダの作成
    os.makedirs(output_folder, exist_ok=True)
    
    # 新世代の個体を保存
    for i, solution in enumerate(new_solutions):
        output_file = os.path.join(output_folder, f"individual_{i+1:03d}.txt")
        write_individual_to_file(solution, output_file)
    
    print(f"✓ 第{generation_num+1}世代（20個体）を {output_folder} に保存しました")
    print(f"CMAESパラメータ: σ={cmaes.sigma:.4f}")
    print(f"現在の平均適応度: {np.mean(fitness_values[valid_mask]):.4f}")
    # total_reflections を制約違反の近似値として出力（従来の constraint_violations は未使用）
    constraint_violations = total_reflections
    print(f"期待される改善: 探索範囲σ={cmaes.sigma:.4f}, 反射による修正箇所={constraint_violations}")
    print("="*60)

if __name__ == "__main__":
    # テスト用のデフォルト設定
    population_folder = "1_generation"
    fitness_filename = "obj_value.csv"
    output_folder = "2_generation"
    generation_num = 1
    
    generate_next_generation_cmaes(population_folder, fitness_filename, output_folder, generation_num)
