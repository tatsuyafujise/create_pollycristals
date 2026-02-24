import cv2
import numpy as np
import os
import ot
from concurrent.futures import ProcessPoolExecutor

def EMD(input_image):
  """Wasserstein距離、Earth Mover's Distanceを計算"""
  try:
    img = cv2.imread(input_image)
    if img is None:
      # print(f"画像読み込み失敗: {input_image}") # 並列時はログが混ざるため抑制
      return np.array([]), np.array([])
      
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 閾値処理の改善（複数の閾値を試行）
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # 輪郭が見つからない場合は閾値を調整して再試行
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
      # より低い閾値で再試行
      _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
      contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
    if len(contours) == 0:
      # さらに低い閾値で再試行
      _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
      contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    weights = []
    total_area = 0
    
    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area < 5:  # 最小面積閾値を下げる
          continue

      M = cv2.moments(cnt)
      if M["m00"] == 0:
          continue

      cx = int(M["m10"] / M["m00"])
      cy = int(M["m01"] / M["m00"])

      points.append([cx, cy])  # リストとして追加
      weights.append(area)
      total_area += area
      # cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1) # 画像保存しないなら描画不要
      # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
    
    # 特徴点が見つからない場合の処理
    if len(points) == 0:
      # 画像の中心点を代表点として使用
      h, w = gray.shape
      points = [[w//2, h//2]]
      weights = [1.0]
      # print(f"Warning: 特徴点が見つからないため、中心点を使用: {input_image}")
    else:
      points = np.array(points)
      weights = np.array(weights)
      if total_area > 0:
        weights = weights / total_area  # 正規化
      else:
        weights = weights / len(weights)  # 面積が0の場合は均等重み

    return np.array(points), np.array(weights)
    
  except Exception as e:
    print(f"EMD処理エラー ({input_image}): {e}")
    return np.array([]), np.array([])
  
  return points, weights

def process_single_image_emd(args):
    """
    並列処理用に1画像の計算を行う関数
    args: (index, generation_num, img2_dir, points_ori, weights_ori)
    """
    i, generation_num, img2_dir, points_ori, weights_ori = args
    
    img2_name_template = f"{generation_num}_individual_{i:03d}.png"
    img2_path = os.path.join(img2_dir, img2_name_template)
    image_name = f"img{i}"
    
    if not os.path.exists(img2_path):
        return image_name, float('nan'), f"Error: Unable to read {img2_path}. Skipping."

    try:
        points_new, weights_new = EMD(img2_path)
        
        # 特徴点が抽出できない場合の処理
        if len(points_new) == 0 or len(weights_new) == 0:
            return image_name, 100.0, f"Warning: {img2_path} から特徴点が抽出できません。ペナルティ値を設定。"
        
        # EMD計算
        M = ot.dist(points_ori, points_new, metric='euclidean')
        emd_value = ot.emd2(weights_ori, weights_new, M)
        
        # 特徴点数の差を計算
        point_count_diff = abs(len(points_new) - len(points_ori))
        
        # 動的重み付けの計算
        if emd_value <= 10.0:
            w = 2.0  # 最適化が進んだ段階では特徴点数の制約を厳密に
        else:
            w = 0.5  # 探索序盤は小さめの重みで自由な探索を許可
        
        # 総合的な適応度値の計算
        fitness_value = emd_value + w * point_count_diff
        
        # 異常値チェック
        if np.isnan(emd_value) or np.isinf(emd_value) or emd_value < 0:
            return image_name, 100.0, f"Warning: {img2_path} で異常なEMD値 {emd_value}。ペナルティ値を設定。"
        
        # 成功時のログメッセージ作成
        log_msg = (f"img{i}: EMD = {emd_value:.4f}, 特徴点数差 = {point_count_diff}, "
                   f"w = {w:.1f}, 総合適応度 = {fitness_value:.4f}")
        return image_name, fitness_value, log_msg
        
    except Exception as e:
        return image_name, 100.0, f"Error: {img2_path} の処理中にエラー: {e}"

def calculate_emd(generation_num):
  """emdの具体的な値を計算（並列化版）"""
  img1_path = "./figure1_2_edge.png"
  if not os.path.exists(img1_path):
    print(f"Error: Unable to read base image {img1_path}. Check the file path.")
    return None, None

  # この img2_dir は run_cmaes_algorithm.py によって置換される想定
  img2_dir = f"./{generation_num}_top_img" # 置換対象のプレースホルダー

  # 基準画像の処理（一度だけ実行）
  try:
    points_ori, weights_ori = EMD(img1_path)
    if len(points_ori) == 0 or len(weights_ori) == 0:
      print(f"Error: 基準画像 {img1_path} から有効な特徴点が抽出できません")
      return None, None
    print(f"基準画像の特徴点数: {len(points_ori)}")
  except Exception as e:
    print(f"Error: 基準画像の処理に失敗: {e}")
    return None, None

  fitness_values = []
  image_names_for_header = []
  
  # 並列処理のためのタスクリスト作成
  tasks = []
  for i in range(1, 21):  # 個体数
      tasks.append((i, generation_num, img2_dir, points_ori, weights_ori))

  print(f"並列処理でEMD計算を開始します (max 10 cores)...")

  # ProcessPoolExecutorによる並列実行
  with ProcessPoolExecutor(max_workers=6) as executor:
      # mapの結果は入力順（img1...img20）に返ってくることが保証される
      results = executor.map(process_single_image_emd, tasks)
      
      for image_name, fitness, log_msg in results:
          image_names_for_header.append(image_name)
          fitness_values.append(fitness)
          if log_msg:
              print(log_msg)

  return image_names_for_header, fitness_values

def main():
  print("Testing objective_func.py directly...")
  generation_to_test = 1 
  
  test_img_dir = f"./{generation_to_test}_top_img"
  os.makedirs(test_img_dir, exist_ok=True)
  
  # calculate_fitness_values を呼び出す前に、img2_dir をテスト用の値に書き換える
  original_content = ""
  with open(__file__, 'r', encoding='utf-8') as f:
    original_content = f.read()
  
  temp_content = original_content.replace(
    'img2_dir = r"CMAES_results/top_img_files/1_top_img" # 第1世代用に自動置換されました',
    f'img2_dir = r"{test_img_dir}" # テスト用のパスに置換'
  )
  with open(__file__, 'w', encoding='utf-8') as f:
    f.write(temp_content)
  
  import importlib
  importlib.reload(objective_func) 

  image_names, fitness_values = objective_func.calculate_emd(generation_to_test)
  
  with open(__file__, 'w', encoding='utf-8') as f:
    f.write(original_content)
  importlib.reload(objective_func)

  if image_names and fitness_values:
    print("\nTest Calculation Summary:")
    print(f"Successfully calculated {len(fitness_values)} fitness values for generation {generation_to_test}.")
  else:
    print("Test calculation failed.")

if __name__ == "__main__":
  main()