import re
import numpy as np
from pathlib import Path
import os

def extract_vertices_with_z_from_tess(file_path, z_value=4.0):
    """TESSファイルから指定されたz座標を持つvertexを抽出（端点を除く）"""
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
                                'index': idx,
                                'x': x,
                                'y': y,
                                'z': z
                            })
                except ValueError:
                    continue
    
    return vertices

def extract_vertices_with_z_from_obj(file_path, z_value=4.0):
    """OBJファイルから指定されたz座標を持つvertexを抽出（端点を除く）"""
    vertices = []
    corner_points = {(8.0, 3.0), (8.0, 6.0), (16.0, 3.0), (16.0, 6.0)}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('v '):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    if abs(z - z_value) < 1e-9:
                        # 端点でない場合のみ追加
                        if (x, y) not in corner_points:
                            vertices.append({
                                'line_num': i,
                                'x': x,
                                'y': y,
                                'z': z,
                                'original_line': line
                            })
                except ValueError:
                    continue
    
    return vertices

def get_edge_type(x, y, tolerance=1e-6):
    """点がどの辺上にあるかを判定（辺上にない場合はNone）"""
    # コーナー点は除外済みなので、辺上の点のみを判定
    if abs(x - 8.0) < tolerance and 3.0 < y < 6.0:
        return 'left'  # x=8.0の辺
    elif abs(x - 16.0) < tolerance and 3.0 < y < 6.0:
        return 'right'  # x=16.0の辺
    elif abs(y - 3.0) < tolerance and 8.0 < x < 16.0:
        return 'bottom'  # y=3.0の辺
    elif abs(y - 6.0) < tolerance and 8.0 < x < 16.0:
        return 'top'  # y=6.0の辺
    return None  # 辺上にない（内部の点）

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

def replace_vertices_in_obj(source_obj_file, reference_tess_file, output_obj_file, verbose=True):
    """OBJファイルのz=4.0のvertex（端点を除く）をTESSファイルの座標に置き換える"""
    
    # TESSファイルからz=4.0のvertexを抽出（参照用、端点を除く）
    reference_vertices = extract_vertices_with_z_from_tess(reference_tess_file)
    
    # OBJファイルからz=4.0のvertexを抽出（端点を除く）
    source_vertices = extract_vertices_with_z_from_obj(source_obj_file)
    
    if verbose:
        print(f"Source OBJ file: {source_obj_file}")
        print(f"  {len(source_vertices)} vertices with z=4.0 (excluding corners)")
        print(f"Reference TESS file: {reference_tess_file}")
        print(f"  {len(reference_vertices)} vertices with z=4.0 (excluding corners)")
    
    if len(source_vertices) != len(reference_vertices):
        error_msg = f"Warning: Different number of vertices! Source: {len(source_vertices)}, Reference: {len(reference_vertices)} - Skipping file"
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)
    
    # OBJファイルを読み込み
    with open(source_obj_file, 'r') as f:
        lines = f.readlines()
    
    # 各source vertexに対して最も近いreference vertexを見つけて置き換え
    # まず、すべての点について辺チェックを行う
    replacements = []
    for source_v in source_vertices:
        closest = find_closest_match(source_v, reference_vertices)
        if closest:
            # 辺上の点かどうかをチェック
            source_edge = get_edge_type(source_v['x'], source_v['y'])
            target_edge = get_edge_type(closest['x'], closest['y'])
            
            # 辺上の点の場合、同じ辺上にあるかチェック
            if source_edge is not None:
                if source_edge != target_edge:
                    error_msg = f"Edge mismatch detected: source point ({source_v['x']:.3f}, {source_v['y']:.3f}) on {source_edge} edge, but closest point ({closest['x']:.3f}, {closest['y']:.3f}) is on {target_edge} edge - Skipping file"
                    if verbose:
                        print(error_msg)
                    raise ValueError(error_msg)
            
            # 頂点行を置き換え
            new_line = f"v {closest['x']:.12f} {closest['y']:.12f} {closest['z']:.12f}\n"
            lines[source_v['line_num']] = new_line
            
            replacements.append({
                'old': (source_v['x'], source_v['y']),
                'new': (closest['x'], closest['y']),
                'distance': np.sqrt((source_v['x'] - closest['x'])**2 + 
                                   (source_v['y'] - closest['y'])**2)
            })
    
    # 結果を出力ファイルに書き込み
    with open(output_obj_file, 'w') as f:
        f.writelines(lines)
    
    if verbose:
        # 置き換え情報を表示
        print(f"\nReplaced {len(replacements)} vertices (corners preserved)")
        print(f"Output saved to: {output_obj_file}\n")
    
    return len(replacements)

def process_all_obj_files(obj_base_dir, reference_tess_file, output_base_dir):
    """obj_files配下のすべてのディレクトリ内のobjファイルを処理"""
    
    obj_base_path = Path(obj_base_dir)
    output_base_path = Path(output_base_dir)
    
    # 出力ディレクトリを作成
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # obj_files配下のすべてのディレクトリを取得
    obj_directories = sorted([d for d in obj_base_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(obj_directories)} directories to process")
    print(f"Reference TESS file: {reference_tess_file}")
    print(f"Output directory: {output_base_dir}")
    print(f"Corner points (8.0,3.0), (8.0,6.0), (16.0,3.0), (16.0,6.0) will be preserved\n")
    
    total_files = 0
    processed_files = 0
    
    for obj_dir in obj_directories:
        # ディレクトリ名を取得 (例: "1000_generation")
        dir_name = obj_dir.name
        
        # 対応する出力ディレクトリを作成
        output_dir = output_base_path / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ディレクトリ内のすべてのobjファイルを取得
        obj_files = list(obj_dir.glob("*.obj"))
        
        if obj_files:
            print(f"Processing {dir_name}: {len(obj_files)} files")
            
            for obj_file in obj_files:
                # 出力ファイル名を生成
                output_file = output_dir / f"{obj_file.stem}_replaced.obj"
                
                try:
                    # 変換処理を実行
                    replace_vertices_in_obj(str(obj_file), reference_tess_file, str(output_file), verbose=False)
                    processed_files += 1
                    total_files += 1
                    
                    if processed_files % 100 == 0:
                        print(f"  Progress: {processed_files} files processed...")
                        
                except Exception as e:
                    print(f"  Error processing {obj_file.name}: {e}")
                    total_files += 1
            
            print(f"  Completed {dir_name}: {len(obj_files)} files processed\n")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {total_files - processed_files}")
    print(f"{'='*60}")

# 実行
if __name__ == "__main__":
    # 設定
    obj_base_dir = r"./good_results/obj_files"
    reference_tess_file = r"./half_of_half.tess"
    output_base_dir = r"./obj_files_replaced"
    
    # バッチ処理を実行
    process_all_obj_files(obj_base_dir, reference_tess_file, output_base_dir)