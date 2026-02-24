import os 
import subprocess
import time
import shutil 
import csv
import objective_func
import importlib
import numpy as np
import re  # 正規表現を使うために追加

# 保存する世代数の設定
KEEP_GENERATIONS_FROM = 4500  # この世代以降のファイルのみ保存

def delete_old_generation_files(generation_num, base_output_dir):
  """指定された世代のファイルを削除する"""
  if generation_num >= KEEP_GENERATIONS_FROM:
    return  # 保存対象世代は削除しない
  
  print(f"第{generation_num}世代のファイルを削除中...")
  
  # 削除するディレクトリのパスリスト
  dirs_to_clean = [
    os.path.join(base_output_dir, "generation_files", f"{generation_num}_generation"),
    os.path.join(base_output_dir, "tess_files", f"{generation_num}_tess"),
    os.path.join(base_output_dir, "png_files", f"{generation_num}_png"),
    os.path.join(base_output_dir, "top_img_files", f"{generation_num}_top_img")
  ]
  
  deleted_count = 0
  for dir_path in dirs_to_clean:
    if os.path.exists(dir_path):
      try:
        shutil.rmtree(dir_path)
        deleted_count += 1
        print(f"  削除: {dir_path}")
      except Exception as e:
        print(f"  削除失敗: {dir_path} - {e}")
  
  if deleted_count > 0:
    print(f"✓ 第{generation_num}世代の{deleted_count}個のディレクトリを削除しました")

def run_command(command):
  """コマンドを実行し、結果を表示する"""
  print(f"実行: {command}")
  result = subprocess.run(command, shell=True, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"エラー出力: {result.stderr}")
    raise Exception(f"コマンド実行エラー: {command}")
  return result.stdout

def update_file_content(file_path, replacements):
  """ファイルの内容を置換する"""
  with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

  for old_str, new_str in replacements.items():
    content = content.replace(old_str, new_str)

  with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

def replace_line_in_file(file_path, line_pattern, new_line):
  """ファイル内の特定のパターンを含む行を置換する（より正確な置換）"""
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  
  found = False
  for i, line in enumerate(lines):
    if line_pattern in line:
      # 元の行のインデントを保持
      indent = len(line) - len(line.lstrip())
      clean_new_line = new_line.lstrip()  # インデントを削除してから再適用
      lines[i] = ' ' * indent + clean_new_line + '\n'
      found = True
      break
  
  if found:
    with open(file_path, 'w', encoding='utf-8') as f:
      f.writelines(lines)
    return True
  return False

def update_generation_number_in_file(file_path, current_gen):
    """ファイル内の世代番号を正規表現を使って更新する"""
    try:
        # ファイルを読み込む
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 変更があったかの確認フラグ
        changes_made = False
        original_content = content
        
        # 正規表現で各パターンを置換
        patterns = [
            # tess_dir の行
            (r'tess_dir = r"CMAES_results/tess_files/\d+_tess"', f'tess_dir = r"CMAES_results/tess_files/{current_gen}_tess"'),
            # png_dir の行
            (r'png_dir = r"CMAES_results/png_files/\d+_png"', f'png_dir = r"CMAES_results/png_files/{current_gen}_png"'),
            # top_img_dir の行
            (r'top_img_dir = r"CMAES_results/top_img_files/\d+_top_img"', f'top_img_dir = r"CMAES_results/top_img_files/{current_gen}_top_img"'),
            # output_name_base の行
            (r'output_name_base = f"\d+_individual_\{file_num\}"', f'output_name_base = f"{current_gen}_individual_{{file_num}}"'),
            # input_file の行（世代に関する部分）
            (r'input_file = f"CMAES_results/generation_files/\d+_generation/individual_\{file_num\}\.txt"', f'input_file = f"CMAES_results/generation_files/{current_gen}_generation/individual_{{file_num}}.txt"'),
        ]
        
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                changes_made = True
        
        # 変更を書き込む
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ {file_path} の世代番号を {current_gen} に更新しました")
            return True
        else:
            print(f"! {file_path} に世代番号の更新対象が見つかりませんでした")
            # デバッグ用：現在のファイル内容の一部を表示
            lines = content.split('\n')
            relevant_lines = [line for line in lines if any(keyword in line for keyword in ['tess_dir', 'png_dir', 'top_img_dir', 'output_name_base', 'input_file'])]
            if relevant_lines:
                print(f"  現在のファイル内容（関連行）:")
                for line in relevant_lines[:10]:  # 最初の10行だけ表示
                    print(f"    {line}")
            return False
    
    except Exception as e:
        print(f"✗ ファイル内の世代番号の更新に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_path_accessible(src_path, link_path):
  """指定されたパスにアクセス可能なようにする（シンボリックリンクまたはディレクトリコピー）"""
  if not os.path.exists(src_path):
    print(f"警告: ソースパス {src_path} が存在しません")
    return False
  
  if os.path.exists(link_path):
    # 既存のものを削除
    try:
      if os.path.islink(link_path):
        os.unlink(link_path)
      elif os.path.isdir(link_path):
        shutil.rmtree(link_path)
      else:
        os.remove(link_path)
    except Exception as e:
      print(f"既存のパス削除に失敗: {e}")
      return False
  
  try:
    # シンボリックリンク作成を試みる（WindowsやWSLでは権限の問題で失敗する可能性あり）
    os.symlink(src_path, link_path)
    print(f"✓ シンボリックリンクを作成: {link_path} -> {src_path}")
    return True
  except Exception as e:
    print(f"シンボリックリンク作成に失敗: {e}")
    try:
      # 代替手段としてディレクトリをコピー
      if os.path.isdir(src_path):
        shutil.copytree(src_path, link_path)
      else:
        shutil.copy2(src_path, link_path)
      print(f"✓ 代替手段としてコピーを作成: {src_path} -> {link_path}")
      return True
    except Exception as e2:
      print(f"コピー作成にも失敗: {e2}")
      return False

def make_backup(file_path):
  """ファイルのバックアップを作成"""
  backup_path = f"{file_path}.backup"
  if os.path.exists(file_path):
    shutil.copy2(file_path, backup_path)
    print(f"バックアップを作成: {backup_path}")
  else:
    print(f"警告: バックアップ対象ファイル {file_path} が存在しません。")

def append_fitness_to_csv(csv_path, generation_num, image_names, fitness_values):
  """適応度csvファイルに新しい世代のデータ行を追加する"""
  file_exists = os.path.exists(csv_path)

  # fitness_valuesの値を小数第3位までに整形（NaN値も適切に処理）
  formatted_fitness_values = []
  for val in fitness_values:
    if isinstance(val, float):
      if np.isnan(val):
        formatted_fitness_values.append("100.000")  # NaNの場合はペナルティ値
      elif np.isinf(val):
        formatted_fitness_values.append("100.000")  # 無限大の場合もペナルティ値
      else:
        formatted_fitness_values.append(f"{val:.3f}")
    else:
      formatted_fitness_values.append(str(val))

  with open(csv_path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow([""] + image_names)

    # 新しい世代の適応度データを書き込む (整形済みの値を使用)
    writer.writerow([f"第{generation_num}世代"] + formatted_fitness_values)

  if not file_exists:
    print(f"CSVファイル {csv_path} を新規作成し、第{generation_num}世代のデータを追加しました。")
  else:
    print(f"第{generation_num}世代のデータを {csv_path} に追加しました。")

def check_and_fix_missing_files(generation_num, top_img_dir):
  """生成されなかったファイルをチェックし、ダミーファイルで補完"""
  missing_files = []
  
  for i in range(1, 21):  # 個体数
    expected_file = os.path.join(top_img_dir, f"{generation_num}_individual_{i:03d}.png")
    if not os.path.exists(expected_file):
      missing_files.append((i, expected_file))
  
  if missing_files:
    print(f"警告: {len(missing_files)}個のファイルが見つかりません")
    
    try:
      import cv2
      import numpy as np
      
      for idx, missing_file in missing_files:
        print(f"ダミーファイルを生成: {missing_file}")
        # 基本的な黒い画像を生成
        dummy_img = np.zeros((216, 576, 3), dtype=np.uint8)
        cv2.imwrite(missing_file, dummy_img)
        
    except Exception as e:
      print(f"ダミーファイル生成エラー: {e}")
  else:
    print(f"✓ 全ての個体ファイルが正常に生成されました")
  
  return len(missing_files)

# メインの制御関数
def run_cmaes_algorithm(start_gen=1, max_gen=5000):
  print(f"CMA-ESアルゴリズムを開始します（{start_gen}世代目〜{max_gen}世代目）")
  
  # 全ての出力を格納する親ディレクトリ
  base_output_dir = "CMAES_results"
  os.makedirs(base_output_dir, exist_ok=True)

  # 各種サブディレクトリを作成
  tess_files_dir = os.path.join(base_output_dir, "tess_files")
  png_files_dir = os.path.join(base_output_dir, "png_files")
  top_img_files_dir = os.path.join(base_output_dir, "top_img_files")
  generation_files_dir = os.path.join(base_output_dir, "generation_files")
  
  # 各サブディレクトリを作成
  os.makedirs(tess_files_dir, exist_ok=True)
  os.makedirs(png_files_dir, exist_ok=True)
  os.makedirs(top_img_files_dir, exist_ok=True)
  os.makedirs(generation_files_dir, exist_ok=True)

  # 各スクリプトのパス
  create_initial_pop_script = "create_initial_population.py"
  generate_tess_script = "generate_tess_png.py"
  generate_next_script = "generate_next_generation_cmaes.py"
    
  # CSVファイルのパス（結果フォルダ内）
  csv_output_file = os.path.join(base_output_dir, "obj_value.csv")

  # 主要スクリプトのバックアップを作成
  make_backup(create_initial_pop_script)
  make_backup(generate_tess_script)
  make_backup(generate_next_script)
  make_backup("objective_func.py") 
    
  # 初期世代の生成（一番最初の世代のみ）
  if start_gen == 1:
    print("\n=== 初期個体群の生成（第1世代） ===")
    # 第1世代の個体群ディレクトリ
    first_gen_pop_dir = os.path.join(generation_files_dir, "1_generation")
    os.makedirs(first_gen_pop_dir, exist_ok=True)
    
    # create_initial_population.pyの出力先を変更
    replacements_create_pop = {
      'SAVE_DIR = "1_generation"': f'SAVE_DIR = r"{first_gen_pop_dir}"'
    }
    update_file_content(create_initial_pop_script, replacements_create_pop)
    
    # 初期個体群生成スクリプトを実行
    run_command(f"python3 {create_initial_pop_script}")
    
    # 元のパスに戻す（他の用途でスクリプトが使われる場合のため）
    replacements_create_pop_revert = {
      f'SAVE_DIR = r"{first_gen_pop_dir}"': 'SAVE_DIR = "1_generation"'
    }
    update_file_content(create_initial_pop_script, replacements_create_pop_revert)

  # CMA-ES状態ファイルをクリア（新規実行時）
  if start_gen == 1:
    cmaes_state_file = os.path.join(base_output_dir, "cmaes_state.pkl")
    if os.path.exists(cmaes_state_file):
      os.remove(cmaes_state_file)
      print("前回のCMA-ES状態ファイルをクリアしました")

  # 各世代のループ処理
  for gen in range(start_gen, max_gen + 1):
    print(f"\n=== 第{gen}世代の処理を開始 ===")

    # --- 世代ごとのディレクトリパス設定 ---
    # 個体群ファイル (例: CMAES_results/generation_files/1_generation)
    current_gen_individual_dir = os.path.join(generation_files_dir, f"{gen}_generation")
    
    # TESSファイル (例: CMAES_results/tess_files/1_tess)
    current_tess_dir = os.path.join(tess_files_dir, f"{gen}_tess")
    
    # PNGファイル (例: CMAES_results/png_files/1_png)
    current_png_dir = os.path.join(png_files_dir, f"{gen}_png")
    
    # Top Imageファイル (例: CMAES_results/top_img_files/1_top_img)
    current_top_img_dir = os.path.join(top_img_files_dir, f"{gen}_top_img")

    # 各ディレクトリを作成
    os.makedirs(current_tess_dir, exist_ok=True)
    os.makedirs(current_png_dir, exist_ok=True)
    os.makedirs(current_top_img_dir, exist_ok=True)
    
    # --- 1. TESSファイルとPNGファイルの生成 ---
    print(f"\n--- TESSファイルとPNGファイルの生成（第{gen}世代） ---")
    
    # generate_tess_png.pyに世代番号を引数として渡して実行
    run_command(f"python3 {generate_tess_script} {gen}")

    print("\n--- 生成されたファイルの検証と補完 ---")
    # 欠損ファイルをチェックして補完
    missing_count = check_and_fix_missing_files(gen, current_top_img_dir)
    # サンプルファイルをチェック
    sample_file = os.path.join(current_top_img_dir, f"{gen}_individual_001.png")
    if os.path.exists(sample_file):
      file_time = time.ctime(os.path.getmtime(sample_file))
      print(f"✓ 第{gen}世代のサンプルファイルが存在します: {sample_file}")
      print(f"  最終更新: {file_time}")
    else:
      print(f"✗ 第{gen}世代のサンプルファイルが見つかりません")
      # 何があるか確認
      if os.path.exists(current_top_img_dir):
        files = os.listdir(current_top_img_dir)[:5]
        print(f"  ディレクトリ内の最初の数ファイル: {files}")

    # --- 2. 適応度計算 ---
    print(f"\n--- 適応度計算（第{gen}世代） ---")
    
    # 実行前のパス構造の確認（ここに追加）
    print(f"--- パス構造の確認 ---")
    print(f"現在の世代: {gen}")
    print(f"カレントディレクトリ: {os.getcwd()}")
    print(f"上面画像ディレクトリ: {current_top_img_dir}")
    print(f"このディレクトリは存在する?: {os.path.exists(current_top_img_dir)}")

    # サンプル画像のパスチェック
    sample_img_path = os.path.join(current_top_img_dir, f"{gen}_individual_001.png")
    print(f"サンプル画像パス: {sample_img_path}")
    print(f"このファイルは存在する?: {os.path.exists(sample_img_path)}")

    # ディレクトリ内のファイル一覧を表示
    if os.path.exists(current_top_img_dir):
      files = os.listdir(current_top_img_dir)
      print(f"ディレクトリ内の最初の5ファイル: {files[:5] if files else '空ディレクトリ'}")
    
    # パス問題を解決するためのシンボリックリンク（またはディレクトリコピー）
    local_top_img_dir = f"./{gen}_top_img"
    ensure_path_accessible(current_top_img_dir, local_top_img_dir)
    print(f"ローカルアクセスパス確認: {os.path.exists(local_top_img_dir)}")
    if os.path.exists(local_top_img_dir):
      local_files = os.listdir(local_top_img_dir)
      print(f"ローカルディレクトリ内のファイル: {local_files[:5] if local_files else '空ディレクトリ'}")

    # objective_func.py の画像パスを置換（行単位の置換）
    search_pattern = "img2_dir = f"
    new_line = f'img2_dir = r"{current_top_img_dir}" # 第{gen}世代用に自動置換されました'
    path_updated = replace_line_in_file("objective_func.py", search_pattern, new_line)
    
    if not path_updated:
      print("警告: objective_func.pyの画像パスの置換に失敗しました")
      # 従来の置換方法をバックアップとして試す
      replacements_obj_func = {
        'img2_dir = f"./{generation_num}_top_img" # 置換対象のプレースホルダー': 
        f'img2_dir = r"{current_top_img_dir}" # 第{gen}世代用に自動置換されました',
      }
      update_file_content("objective_func.py", replacements_obj_func)

    # 置換後の検証（ここに追加）
    print(f"--- objective_func.pyの置換後の検証 ---")
    with open("objective_func.py", 'r', encoding='utf-8') as f:
      content = f.read()
      img2_dir_lines = [line for line in content.split('\n') if 'img2_dir =' in line]
      if img2_dir_lines:
        print(f"置換後の設定: {img2_dir_lines[0]}")
        if current_top_img_dir in img2_dir_lines[0]:
          print(f"✓ 置換成功: パスが正しく更新されました")
        else:
          print(f"✗ 置換失敗: 正しいパスに更新されていません")
      else:
        print("警告: img2_dir設定が見つかりません")
    
    # モジュールを再読み込みして変更を反映
    importlib.reload(objective_func)

    # 適応度計算実行
    try:
      # image_names, fitness_values = objective_func.calculate_fitness_values(gen)
      image_names, fitness_values = objective_func.calculate_emd(gen)
      if image_names is None or fitness_values is None:
        print(f"エラー: 第{gen}世代の適応度計算に失敗しました。処理を中断します。")
        return
      
      # 適応度の要約情報を表示
      valid_fitness = [f for f in fitness_values if isinstance(f, (int, float)) and not np.isnan(f) and not np.isinf(f)]

      current_best_fitness = float('inf')

      if valid_fitness:
        min_fitness = min(valid_fitness)
        current_best_fitness = min_fitness
        min_idx = next(i for i, f in enumerate(fitness_values) if f == min_fitness)
        max_fitness = max(valid_fitness)
        avg_fitness = sum(valid_fitness) / len(valid_fitness)
        print(f"\n【第{gen}世代の適応度情報】")
        print(f"  最小値: {min_fitness:.4f} (最良個体: {image_names[min_idx]})")
        print(f"  平均値: {avg_fitness:.4f}")
        print(f"  最大値: {max_fitness:.4f}")

        if current_best_fitness <= 0.6:
          print(f"\n目標達成: 適応度が0.6以下に到達しました")        
          # 最後のデータもCSVに記録してから終了
          append_fitness_to_csv(csv_output_file, gen, image_names, fitness_values)
          print(f"第{gen}世代で最適化を正常終了します。")
          break
          
      else:
        print(f"\n【第{gen}世代の適応度情報】")
        print(f"  警告: 有効な適応度値が見つかりませんでした")
      
      # 適応度データをCSVに追加
      append_fitness_to_csv(csv_output_file, gen, image_names, fitness_values)
    except Exception as e:
      print(f"適応度計算中にエラーが発生しました: {e}")
      import traceback
      traceback.print_exc()
      return
    
    # 作成したシンボリックリンクを削除
    if os.path.exists(local_top_img_dir):
      try:
        if os.path.islink(local_top_img_dir):
          os.unlink(local_top_img_dir)
        elif os.path.isdir(local_top_img_dir):
          shutil.rmtree(local_top_img_dir)
        print(f"✓ 一時的なリンク/フォルダを削除しました: {local_top_img_dir}")
      except Exception as e:
        print(f"リンク/フォルダの削除に失敗: {e}")
    
    # objective_func.py を元に戻す（行単位の置換）
    original_line = 'img2_dir = f"./{generation_num}_top_img" # 置換対象のプレースホルダー'
    replace_line_in_file("objective_func.py", "img2_dir =", original_line)

    # --- 3. 2世代前のファイルを削除 ---
    # 4500世代未満で、かつ2世代以上進んでいる場合に削除
    gen_to_delete = gen - 2
    if gen_to_delete >= 1 and gen_to_delete < KEEP_GENERATIONS_FROM:
      print(f"\n--- 2世代前のファイル削除（第{gen_to_delete}世代） ---")
      delete_old_generation_files(gen_to_delete, base_output_dir)

    # 最終世代なら終了
    if gen == max_gen:
      print(f"\n=== CMA-ESアルゴリズムが完了しました（全{max_gen}世代） ===")
      break
    
    # --- 4. 次世代の生成（CMA-ES使用） ---
    print(f"\n--- 次世代の生成（第{gen+1}世代、CMA-ES） ---")
    # 次世代の個体群ディレクトリ
    next_gen_individual_dir = os.path.join(generation_files_dir, f"{gen+1}_generation")
    os.makedirs(next_gen_individual_dir, exist_ok=True)

    # generate_next_generation_cmaes.py のパス置換
    replacements_next_gen = {
      # 入力フォルダ（現在の世代）
      'population_folder = "1_generation"': f'population_folder = r"{current_gen_individual_dir}"',
      # 適応度ファイル
      'fitness_filename = "obj_value.csv"': f'fitness_filename = r"{csv_output_file}"',
      # 出力フォルダ（次の世代）
      'output_folder = "2_generation"': f'output_folder = r"{next_gen_individual_dir}"',
      # 世代番号
      'generation_num = 1': f'generation_num = {gen}'
    }
    update_file_content(generate_next_script, replacements_next_gen)
    
    # 次世代生成スクリプトを実行
    run_command(f"python3 {generate_next_script}")
    
    # スクリプトの設定を元に戻す(オプション)
    replacements_next_gen_revert = {
      f'population_folder = r"{current_gen_individual_dir}"': 'population_folder = "1_generation"',
      f'fitness_filename = r"{csv_output_file}"': 'fitness_filename = "obj_value.csv"',
      f'output_folder = r"{next_gen_individual_dir}"': 'output_folder = "2_generation"',
      f'generation_num = {gen}': 'generation_num = 1'
    }
    update_file_content(generate_next_script, replacements_next_gen_revert)

  print("\n全ての処理が完了しました。")
  print(f"結果ファイルは {base_output_dir} ディレクトリに保存されています。")

if __name__ == "__main__":  # 実行設定
  START_GENERATION = 1  # 開始世代
  MAX_GENERATIONS = 5000  # 最大世代数（3500世代実行）

  # 実行前にスクリプトを初期状態に戻す関数
  def reset_scripts():
    """バックアップファイルからスクリプトを復元する"""
    scripts = ["objective_func.py", "generate_tess_png.py", "generate_next_generation_cmaes.py", "create_initial_population.py"]
    for script in scripts:
      backup_file = f"{script}.backup"
      if os.path.exists(backup_file):
        print(f"スクリプト {script} を初期状態に復元中...")
        shutil.copy2(backup_file, script)
    print("スクリプトを初期状態に復元しました")
  
  # 実行前に世代番号が1になっていることを確認する関数
  def ensure_init_generation_one():
    """スクリプトファイルの世代番号を初期化する"""
    print("スクリプトの世代番号を初期化中...")
    update_generation_number_in_file("generate_tess_png.py", 1)
    
    # 強制的に第1世代に設定（念のため）
    with open("generate_tess_png.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 直接的な置換を実行
    replacements = {
        'tess_dir = r"CMAES_results/tess_files/': 'tess_dir = r"CMAES_results/tess_files/1_tess"  # ',
        'png_dir = r"CMAES_results/png_files/': 'png_dir = r"CMAES_results/png_files/1_png"  # ',
        'top_img_dir = r"CMAES_results/top_img_files/': 'top_img_dir = r"CMAES_results/top_img_files/1_top_img"  # ',
    }
    
    # 各置換を実行
    modified = False
    for pattern_start, replacement in replacements.items():
        if pattern_start in content:
            # 既存の行を見つけて置換
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if pattern_start in line and '_tess"' in line or '_png"' in line or '_top_img"' in line:
                    if '_tess"' in line:
                        lines[i] = 'tess_dir = r"CMAES_results/tess_files/1_tess"'
                    elif '_png"' in line:
                        lines[i] = 'png_dir = r"CMAES_results/png_files/1_png"'
                    elif '_top_img"' in line:
                        lines[i] = 'top_img_dir = r"CMAES_results/top_img_files/1_top_img"'
                    modified = True
                    break
            content = '\n'.join(lines)
    
    # output_name_baseとinput_fileも強制的に第1世代に設定
    import re
    content = re.sub(r'output_name_base = f"\d+_individual_', 'output_name_base = f"1_individual_', content)
    content = re.sub(r'input_file = f"CMAES_results/generation_files/\d+_generation/', 'input_file = f"CMAES_results/generation_files/1_generation/', content)
    
    if modified or 'output_name_base = f"1_individual_' not in content:
        with open("generate_tess_png.py", 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ 強制的な世代番号初期化を実施しました")
    
    print("世代番号の初期化完了")
  
  # スクリプトをリセットと世代番号初期化
  reset_scripts()
  ensure_init_generation_one()

  try:
    print("CMA-ES実行開始...")
    # CMA-ESアルゴリズム実行
    run_cmaes_algorithm(START_GENERATION, MAX_GENERATIONS)
  except Exception as e:
    import traceback
    print(f"エラーが発生しました: {e}")
    print("詳細なエラー情報:")
    traceback.print_exc()
