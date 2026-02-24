import os
import re

def convert_obj_file(input_filename, output_dir):
    """単一OBJファイルを複数の個別OBJファイルに分割する関数"""
    print(f"処理中: {input_filename}")
    
    with open(input_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 全ての頂点情報をリストに保存
    vertices = [line.strip() for line in lines if line.startswith("v ")]

    # グループごとの面情報を辞書に集める
    group_faces = {}
    current_group = None
    for line in lines:
        line = line.strip()
        if line.startswith("g "):
            # グループ名を取得
            parts = line.split()
            if len(parts) >= 2:
                current_group = parts[1]
                group_faces[current_group] = []
        elif line.startswith("f ") and current_group is not None:
            group_faces[current_group].append(line)
            
    # ディレクトリが存在しなければ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 各グループごとにobjファイルを作成
    created_files = []
    for group, faces in group_faces.items():
        if not faces:
            continue
            
        output_filename = os.path.join(output_dir, f"{group}.obj")
        created_files.append(output_filename)
        
        with open(output_filename, "w", encoding="utf-8") as f:
            # 頂点情報を書き込み
            for vertex in vertices:
                f.write(vertex + "\n")
            
            # このグループの面情報を書き込み
            for face in faces:
                f.write(face + "\n")
    
    return created_files

# 現在のスクリプトのディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

# メインディレクトリの作成
main_output_dir = "each_crystal"
os.makedirs(main_output_dir, exist_ok=True)

# 現在のディレクトリ内のすべてのOBJファイルを処理
processed_count = 0

for filename in os.listdir(script_dir):
    if filename.endswith(".obj"):
        print(f"\n処理中: {filename}")
        
        # ファイル名から番号を抽出 (例: individual_001.obj から 001 を取得)
        match = re.search(r'individual_(\d{3})\.obj', filename)
        if match:
            file_num = match.group(1)
        else:
            # ファイル名から番号が取得できない場合は、ファイル名（拡張子なし）をそのまま使用
            file_num = os.path.splitext(filename)[0]
        
        input_path = os.path.join(script_dir, filename)
        output_dir = os.path.join(main_output_dir, file_num)
        
        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 変換処理実行
        created_files = convert_obj_file(input_path, output_dir)
        processed_count += 1
        
        print(f"ファイル {filename} から {len(created_files)} 個のオブジェクトを抽出しました")
        print(f"保存先: {output_dir}/")

print(f"\n処理完了: {processed_count} ファイルを処理しました")