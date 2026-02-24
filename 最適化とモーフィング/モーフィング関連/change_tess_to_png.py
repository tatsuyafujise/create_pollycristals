import os
import subprocess
import time
from pathlib import Path

# カレントディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# tess_files_replaced配下のすべてのディレクトリからtessファイルを検索
tess_base_dir = os.path.join(current_dir, ".", "tess_files_replaced")

if not os.path.exists(tess_base_dir):
    print(f"エラー: {tess_base_dir} ディレクトリが見つかりません")
    exit(1)

# tess_files_replaced配下のすべてのtessファイルを再帰的に検索
tess_files = []
tess_base_path = Path(tess_base_dir)
for tess_dir in sorted(tess_base_path.iterdir()):
    if tess_dir.is_dir():
        tess_files.extend(list(tess_dir.glob("*.tess")))

if not tess_files:
    print("エラー: tess_files_replaced配下にtessファイルが見つかりません")
    exit(1)

print(f"見つかったTESSファイル: {len(tess_files)}個")

# 出力ディレクトリを作成
png_base_dir = os.path.join(current_dir, ".", "png_files_replaced")
top_png_base_dir = os.path.join(current_dir, ".", "top_png_files_replaced")
os.makedirs(png_base_dir, exist_ok=True)
os.makedirs(top_png_base_dir, exist_ok=True)

# 処理カウンタの初期化
total_files = len(tess_files)
error_count = 0
success_count = 0

print(f"PNG、上面画像の生成を開始します...")
print(f"出力先 - PNG: {png_base_dir}")
print(f"出力先 - 上面PNG: {top_png_base_dir}\n")

# 各TESSファイルに対して処理を実行
for idx, tess_file in enumerate(tess_files, 1):
    file_basename = tess_file.name
    file_name_without_ext = tess_file.stem
    parent_dir_name = tess_file.parent.name  # 親ディレクトリ名（例: 1354_tess）
    
    print(f"処理中: {parent_dir_name}/{file_basename} ({idx}/{total_files})")
    
    try:
        # 作業ディレクトリに移動
        os.chdir(current_dir)
        
        # 各親ディレクトリに対応する出力ディレクトリを作成
        png_output_dir = os.path.join(png_base_dir, parent_dir_name)
        top_png_output_dir = os.path.join(top_png_base_dir, parent_dir_name)
        os.makedirs(png_output_dir, exist_ok=True)
        os.makedirs(top_png_output_dir, exist_ok=True)
        
        # PNGファイルの生成
        print("  PNGを生成中...")
        png_output = f"{file_name_without_ext}_png"
        cmd2 = f'neper -V {str(tess_file)} -print {png_output}'
        subprocess.run(cmd2, shell=True, check=True)
        
        # 上面視点からの画像を生成
        print("  上面画像を生成中...")
        top_output = f"{file_name_without_ext}_top"
        cmd3 = f'neper -V {str(tess_file)} -showface "z>0.49" -cameracoo "12.0:4.5:21.7" -imagesize 576:216 -print {top_output}'
        subprocess.run(cmd3, shell=True, check=True)
        
        # 生成されたファイルを適切なディレクトリに移動
        if os.path.exists(f"{png_output}.png"):
            os.rename(f"{png_output}.png", os.path.join(png_output_dir, f"{file_name_without_ext}.png"))
        
        if os.path.exists(f"{top_output}.png"):
            os.rename(f"{top_output}.png", os.path.join(top_png_output_dir, f"{file_name_without_ext}.png"))
        
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
print(f"PNGファイルは {png_base_dir} に保存されました。")
print(f"上面視点PNGファイルは {top_png_base_dir} に保存されました。")
print(f"{'='*60}")
