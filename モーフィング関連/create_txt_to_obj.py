import os
import subprocess
import time
from pathlib import Path

# カレントディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# generation_files配下のすべてのディレクトリからtxtファイルを検索
generation_base_dir = os.path.join(current_dir, "good_results", "generation_files")

if not os.path.exists(generation_base_dir):
    print(f"エラー: {generation_base_dir} ディレクトリが見つかりません")
    exit(1)

# generation_files配下のすべてのtxtファイルを再帰的に検索
txt_files = []
generation_base_path = Path(generation_base_dir)
for gen_dir in sorted(generation_base_path.iterdir()):
    if gen_dir.is_dir():
        txt_files.extend(list(gen_dir.glob("*.txt")))

if not txt_files:
    print("エラー: generation_files配下にtxtファイルが見つかりません")
    exit(1)

print(f"見つかったTXTファイル: {len(txt_files)}個")

# 出力ディレクトリを作成
obj_base_dir = os.path.join(current_dir, "good_results", "obj_files")
os.makedirs(obj_base_dir, exist_ok=True)

# 処理カウンタの初期化
total_files = len(txt_files)
error_count = 0
success_count = 0

print(f"OBJファイルの生成を開始します...")
print(f"出力先: {obj_base_dir}\n")

# 各TXTファイルに対して処理を実行
for idx, txt_file in enumerate(txt_files, 1):
    file_basename = txt_file.name
    file_name_without_ext = txt_file.stem
    parent_dir_name = txt_file.parent.name  # 親ディレクトリ名（例: 1000_generation）
    
    print(f"処理中: {parent_dir_name}/{file_basename} ({idx}/{total_files})")
    
    try:
        # 作業ディレクトリに移動
        os.chdir(current_dir)
        
        # 各親ディレクトリに対応する出力ディレクトリを作成
        obj_output_dir = os.path.join(obj_base_dir, parent_dir_name)
        os.makedirs(obj_output_dir, exist_ok=True)
        
        # OBJファイルの生成
        print("  OBJを生成中...")
        temp_obj_name = f"{file_name_without_ext}_temp"
        cmd = f'neper -T -dim 3 -domain "cube(8.0,3.0,4.0):translate(8.0,3.0,0.0)" -n 15 -morphooptiini "coo:file({str(txt_file)})" -o {temp_obj_name} -format obj'
        subprocess.run(cmd, shell=True, check=True)
        
        # 生成されたOBJファイルを適切なディレクトリに移動
        temp_obj_file = f"{temp_obj_name}.obj"
        if os.path.exists(temp_obj_file):
            final_obj_path = os.path.join(obj_output_dir, f"{file_name_without_ext}.obj")
            os.rename(temp_obj_file, final_obj_path)
            print(f"  OBJファイルを保存: {final_obj_path}")
        
        success_count += 1
        
    except Exception as e:
        print(f"エラー発生: {file_basename} - {str(e)}")
        error_count += 1
    
    # 進捗表示
    if idx % 10 == 0 or idx == total_files:
        print(f"進捗: {idx}/{total_files} ({idx/total_files*100:.1f}%) 完了 - 成功: {success_count}, エラー: {error_count}\n")
    
    # システム負荷軽減のため少し待機
    time.sleep(0.1)

print(f"\n{'='*60}")
print(f"処理完了！")
print(f"総ファイル数: {total_files}")
print(f"成功: {success_count}")
print(f"エラー: {error_count}")
print(f"OBJファイルは {obj_base_dir} に保存されました。")
print(f"{'='*60}")
