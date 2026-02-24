#!/bin/bash

# --- Pythonスクリプトの生成 (幾何計算用) ---
cat << 'EOF' > geometry_sorter.py
import os
import re
import sys

def get_grain_info(filename):
    """
    STEPファイルを読み込み、Z=4.0面上の点の有無と、その場合の平均X座標を返す。
    Z=4.0上の点がない場合は、元のファイル番号を返す。
    """
    points_on_top_x = []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # CARTESIAN_POINTの座標を抽出する正規表現
            # 形式例: CARTESIAN_POINT('',(15.45,3.,4.))
            # STEPファイルは科学的表記(1.E-02)を含むためそれにも対応
            matches = re.findall(r"CARTESIAN_POINT\s*\('[^']*',\s*\(\s*([-\d\.E\+]+)\s*,\s*([-\d\.E\+]+)\s*,\s*([-\d\.E\+]+)\s*\)\s*\)", content)
            
            for x_str, y_str, z_str in matches:
                z = float(z_str)
                # 浮動小数点の誤差を考慮して Z=4 か判定
                if abs(z - 4.0) < 0.001:
                    points_on_top_x.append(float(x_str))
                    
    except Exception as e:
        sys.stderr.write(f"Error reading {filename}: {e}\n")
        return None

    # 元のファイル番号を抽出 (例: cell12.step -> 12)
    try:
        original_num = int(re.search(r'cell(\d+)\.step', filename).group(1))
    except:
        original_num = 99999

    is_top_grain = len(points_on_top_x) > 0
    avg_x = sum(points_on_top_x) / len(points_on_top_x) if is_top_grain else 0
    
    return {
        'filename': filename,
        'is_top': is_top_grain,
        'avg_x': avg_x,
        'original_num': original_num
    }

def main():
    # ディレクトリ内のcell*.stepファイルを取得
    files = [f for f in os.listdir('.') if f.startswith('cell') and f.endswith('.step')]
    
    grains_data = []
    for f in files:
        info = get_grain_info(f)
        if info:
            grains_data.append(info)

    # Top面にある粒 (z=4) と そうでない粒に分ける
    top_grains = [g for g in grains_data if g['is_top']]
    other_grains = [g for g in grains_data if not g['is_top']]

    # ソート処理
    # 1. Top面にある粒は X座標 (avg_x) の昇順 (左から右)
    top_grains.sort(key=lambda x: x['avg_x'])
    
    # 2. その他の粒は 元のファイル番号順
    other_grains.sort(key=lambda x: x['original_num'])

    # リネームコマンドの生成
    # まず一時ファイル名に変更して衝突を避ける
    new_mapping = []
    
    # Top grain (1 to 5)
    counter = 1
    for g in top_grains:
        new_name = f"cell{counter}.step"
        print(f"mv {g['filename']} temp_{new_name}")
        new_mapping.append((f"temp_{new_name}", new_name))
        counter += 1
        
    # Other grains (6 to N)
    for g in other_grains:
        new_name = f"cell{counter}.step"
        print(f"mv {g['filename']} temp_{new_name}")
        new_mapping.append((f"temp_{new_name}", new_name))
        counter += 1

    # 最終的な名前に戻すコマンドを出力
    for temp, final in new_mapping:
        print(f"mv {temp} {final}")

if __name__ == "__main__":
    main()
EOF

# --- メイン処理: 全ディレクトリに対して実行 ---

# 現在のディレクトリにある try* ディレクトリを検索
for i in {292..312}; do
    dir="try${i}"

    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        
        # ディレクトリ内に入る
        cd "$dir" || continue
        
        # Pythonスクリプトを親ディレクトリからコピーして実行する代わりに、
        # ここではPythonコードをパイプで渡すか、親のパスを参照します。
        # 簡易化のため、生成したPythonスクリプトを親から参照して実行します。
        python3 ../geometry_sorter.py > rename_commands.sh
        
        # リネームコマンドの実行
        if [ -s rename_commands.sh ]; then
            bash rename_commands.sh
            echo "  -> Renaming complete."
        else
            echo "  -> No step files found or error occurred."
        fi
        
        # 後始末
        rm rename_commands.sh
        
        # 元の場所に戻る
        cd ..
    fi
done

# Pythonスクリプトの削除
rm geometry_sorter.py

echo "All Done!"
