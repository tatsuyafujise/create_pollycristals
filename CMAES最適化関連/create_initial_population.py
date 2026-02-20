import numpy as np 
import os

# パラメータ
POP_SIZE = 20      # 個体群サイズ
NUM_POINTS = 15    # 1個体あたりの頂点数

X_MAX = 24.0
Y_MAX = 9.0
Z_MAX = 8.0

# 保存先ディレクトリ
SAVE_DIR = "1_generation"

# ディレクトリが存在しない場合は作成
if not os.path.exists(SAVE_DIR):
  os.makedirs(SAVE_DIR)

# 初期個体群の生成と保存（CMAES用）
def create_and_save_initial_population():
  for ind_idx in range(POP_SIZE):
    x_coords = np.random.uniform(0, X_MAX, NUM_POINTS)
    y_coords = np.random.uniform(0, Y_MAX, NUM_POINTS)
    z_coords = np.random.uniform(0, Z_MAX, NUM_POINTS)

    # 3つの座標を組み合わせて個体を生成
    individual = np.column_stack((x_coords, y_coords, z_coords))

    # ファイル名の設定 (例: individual_001.txt, individual_002.txt, ...)
    filename = os.path.join(SAVE_DIR, f"individual_{ind_idx+1:03d}.txt")

    # テキストファイルに保存
    with open(filename, 'w') as f:
      for point_idx in range(NUM_POINTS):
        x, y, z = individual[point_idx]
        f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")

    print(f"Saved individual {ind_idx+1} to {filename}")

if __name__ == "__main__":
  create_and_save_initial_population()
