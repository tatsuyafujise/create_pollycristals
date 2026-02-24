import os
import subprocess
import time
import sys
from concurrent.futures import ProcessPoolExecutor

def get_current_generation():
    """現在の世代番号を動的に取得する"""
    try:
        # コマンドライン引数から世代番号を取得（推奨）
        if len(sys.argv) > 1:
            return int(sys.argv[1])
        
        # CMAES_results/generation_files内の最新世代フォルダから取得
        generation_dir = "CMAES_results/generation_files"
        if os.path.exists(generation_dir):
            gen_folders = [f for f in os.listdir(generation_dir) if f.endswith('_generation')]
            if gen_folders:
                # 数字の部分を抽出してソート
                gen_numbers = []
                for folder in gen_folders:
                    try:
                        gen_num = int(folder.split('_')[0])
                        gen_numbers.append(gen_num)
                    except ValueError:
                        continue
                if gen_numbers:
                    return max(gen_numbers)
        
        # デフォルト値
        return 1
    except Exception as e:
        print(f"世代番号取得エラー: {e}")
        return 1

# 現在の世代番号を取得（グローバル変数として定義）
current_gen = get_current_generation()

# ベースとなる出力ディレクトリ
base_tess_dir_name = "tess_files"
base_png_dir_name = "png_files"
base_top_img_dir_name = "top_img_files"

# ディレクトリパスの定義
tess_dir = f"CMAES_results/tess_files/{current_gen}_tess"
png_dir = f"CMAES_results/png_files/{current_gen}_png"
top_img_dir = f"CMAES_results/top_img_files/{current_gen}_top_img"

def process_single_individual(i):
    """
    1個体分の生成処理を行う関数（並列化用）
    """
    file_num = f"{i:03d}" # 001, 002, ..., 020
    input_file = f"CMAES_results/generation_files/{current_gen}_generation/individual_{file_num}.txt"
    output_name_base = f"{current_gen}_individual_{file_num}"

    # Neperが各プロセスで全コアを使わないように環境変数を設定
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    try:
        # 1. TESSファイルの生成
        cmd1 = f"neper -T -dim 3 -domain \"cube(8.0,3.0,4.0):translate(8.0,3.0,0.0)\" -n 15 -morphooptiini \"coo:file({input_file})\" -o {output_name_base}"
        # Python 3.6対応: capture_outputの代わりにDEVNULLを使用
        subprocess.run(cmd1, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. PNGファイルの生成
        cmd2 = f"neper -V {output_name_base}.tess -print {output_name_base}"
        subprocess.run(cmd2, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. 上面(z>0.99)視点からの画像を生成
        top_image_neper_output_name = f"{output_name_base}_top"
        cmd3 = f"neper -V {output_name_base}.tess -showface \"z>0.49\" -cameracoo \"12.0:4.5:21.7\" -imagesize 576:216 -print {top_image_neper_output_name}"
        subprocess.run(cmd3, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 生成されたファイルを適切なディレクトリに移動
        if os.path.exists(f"{output_name_base}.tess"):
            os.rename(f"{output_name_base}.tess", f"{tess_dir}/{output_name_base}.tess")

        if os.path.exists(f"{output_name_base}.png"):
            os.rename(f"{output_name_base}.png", os.path.join(png_dir, f"{output_name_base}.png"))

        # 上面視点の画像をimgディレクトリに移動
        if os.path.exists(f"{top_image_neper_output_name}.png"):
            destination_top_image_name = f"{output_name_base}.png" # _top を除去したファイル名
            os.rename(f"{top_image_neper_output_name}.png", os.path.join(top_img_dir, destination_top_image_name))

        print(f"完了: {output_name_base}")
        return True

    except subprocess.CalledProcessError:
        print(f"Neper実行エラー: {output_name_base}")
        return False
    except Exception as e:
        print(f"エラー発生: {output_name_base} - {str(e)}")
        return False

def main():
    print(f"現在の世代: {current_gen}")

    # ディレクトリを作成（動的世代番号を使用）
    os.makedirs(tess_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(top_img_dir, exist_ok=True)

    # 処理ファイル数
    total_files = 20
    
    print(f"TESSファイルとPNGファイルの生成を開始します (並列処理: max 6 cores)...")
    start_time = time.time()

    # ProcessPoolExecutorを使用して並列処理を実行
    # 12コア中10コアを使用（システム負荷軽減のため）
    with ProcessPoolExecutor(max_workers=6) as executor:
        # mapを使うことで、順次タスクを割り振る
        results = list(executor.map(process_single_individual, range(1, total_files + 1)))

    # 結果の集計
    success_count = results.count(True)
    error_count = results.count(False)
    elapsed_time = time.time() - start_time

    print(f"\n処理完了！ ({elapsed_time:.1f}秒)")
    print(f"総ファイル数: {total_files}")
    print(f"成功: {success_count}")
    print(f"エラー: {error_count}")
    print(f"TESSファイルは {tess_dir}/ に保存されました。")
    print(f"PNGファイルは {png_dir}/ に保存されました。")
    print(f"上面視点PNGファイルは {top_img_dir}/ に保存されました。")

if __name__ == "__main__":
    main()