import re
import numpy as np
from pathlib import Path
import os

def extract_vertices_with_z(file_path, z_value=4.0):
    """指定されたz座標を持つvertexを抽出（端点を除く）"""
    vertices = []
    corner_points = {(8.0, 3.0), (8.0, 6.0), (16.0, 3.0), (16.0, 6.0)}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    in_vertex_section = False
    for i, line in enumerate(lines):
        if '**vertex' in line:
            in_vertex_section = True
            continue
        if in_vertex_section and '**edge' in line:
            break
        if in_vertex_section:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    idx = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    if abs(z - z_value) < 1e-9:
                        # 端点でない場合のみ追加
                        if (x, y) not in corner_points:
                            vertices.append({
                                'line_num': i,
                                'index': idx,
                                'x': x,
                                'y': y,
                                'z': z,
                                'original_line': line
                            })
                except ValueError:
                    continue
    
    return vertices

def find_closest_match(source_vertex, target_vertices):
    """最も近い(x,y)座標を持つvertexを見つける"""
    min_dist = float('inf')
    closest = None
    
    for target in target_vertices:
        dist = np.sqrt((source_vertex['x'] - target['x'])**2 + 
                      (source_vertex['y'] - target['y'])**2)
        if dist < min_dist:
            min_dist = dist
            closest = target
    
    return closest

def replace_vertices(source_file, reference_file, output_file, verbose=True):
    """z=4.0のvertexを最も近い座標に置き換える"""
    
    # 両ファイルからz=4.0のvertexを抽出
    source_vertices = extract_vertices_with_z(source_file)
    reference_vertices = extract_vertices_with_z(reference_file)
    
    if verbose:
        print(f"Source file: {source_file}")
        print(f"  {len(source_vertices)} vertices with z=4.0")
        print(f"Reference file: {reference_file}")
        print(f"  {len(reference_vertices)} vertices with z=4.0")
    
    if len(source_vertices) != len(reference_vertices):
        error_msg = f"Warning: Different number of vertices! Source: {len(source_vertices)}, Reference: {len(reference_vertices)} - Skipping file"
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)
    
    # ファイルを読み込み
    with open(source_file, 'r') as f:
        lines = f.readlines()
    
    # 各source vertexに対して最も近いreference vertexを見つけて置き換え
    replacements = []
    for source_v in source_vertices:
        closest = find_closest_match(source_v, reference_vertices)
        if closest:
            # 元の行のフォーマットを保持して置き換え
            parts = source_v['original_line'].split()
            new_line = f"  {parts[0]}  {closest['x']:.12f} {closest['y']:.12f} {closest['z']:.12f}     {parts[4]}\n"
            lines[source_v['line_num']] = new_line
            
            replacements.append({
                'index': source_v['index'],
                'old': (source_v['x'], source_v['y']),
                'new': (closest['x'], closest['y']),
                'distance': np.sqrt((source_v['x'] - closest['x'])**2 + 
                                   (source_v['y'] - closest['y'])**2)
            })
    
    # 結果を出力ファイルに書き込み
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    if verbose:
        # 置き換え情報を表示
        print(f"\nReplaced {len(replacements)} vertices (corners preserved)")
        print(f"Output saved to: {output_file}\n")
    
    return len(replacements)

def process_all_tess_files(tess_base_dir, reference_file, output_base_dir):
    """tess_files配下のすべてのディレクトリ内のtessファイルを処理"""
    
    tess_base_path = Path(tess_base_dir)
    output_base_path = Path(output_base_dir)
    
    # 出力ディレクトリを作成
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # tess_files配下のすべてのディレクトリを取得
    tess_directories = sorted([d for d in tess_base_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(tess_directories)} directories to process")
    print(f"Reference file: {reference_file}")
    print(f"Output directory: {output_base_dir}")
    print(f"Corner points (8.0,3.0), (8.0,6.0), (16.0,3.0), (16.0,6.0) will be preserved\n")
    
    total_files = 0
    processed_files = 0
    
    for tess_dir in tess_directories:
        # ディレクトリ名を取得 (例: "1354_tess")
        dir_name = tess_dir.name
        
        # 対応する出力ディレクトリを作成
        output_dir = output_base_path / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ディレクトリ内のすべてのtessファイルを取得
        tess_files = list(tess_dir.glob("*.tess"))
        
        if tess_files:
            print(f"Processing {dir_name}: {len(tess_files)} files")
            
            for tess_file in tess_files:
                # 出力ファイル名を生成
                output_file = output_dir / f"{tess_file.stem}_replaced.tess"
                
                try:
                    # 変換処理を実行
                    replace_vertices(str(tess_file), reference_file, str(output_file), verbose=False)
                    processed_files += 1
                    total_files += 1
                    
                    if processed_files % 100 == 0:
                        print(f"  Progress: {processed_files} files processed...")
                        
                except Exception as e:
                    print(f"  Error processing {tess_file.name}: {e}")
                    total_files += 1
            
            print(f"  Completed {dir_name}: {len(tess_files)} files processed\n")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {total_files - processed_files}")
    print(f"{'='*60}")

# 実行
if __name__ == "__main__":
    # 設定
    tess_base_dir = r"./good_results/tess_files"
    reference_file = r"./half_of_half.tess"
    output_base_dir = r"./tess_files_replaced"
    
    # バッチ処理を実行
    process_all_tess_files(tess_base_dir, reference_file, output_base_dir)