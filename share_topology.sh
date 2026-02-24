#!/bin/bash

# ==========================================
# 設定: パスと許容値
# ==========================================
SC_PATH="/mnt/c/Program Files/Ansys Inc/v231/scdm/SpaceClaim.exe"

# 許容値を 0.1mm に設定 (標準的かつ確実な値)
TOLERANCE_MM="0.02"

# ==========================================
# 環境チェック
# ==========================================
if [ ! -f "$SC_PATH" ]; then
    echo "Error: SpaceClaim executable not found."
    exit 1
fi

if command -v wslpath >/dev/null; then
    to_win_path() { wslpath -w "$1"; }
elif command -v cygpath >/dev/null; then
    to_win_path() { cygpath -w "$1"; }
else
    echo "Error: WSL or Git Bash is required."
    exit 1
fi

BASE_DIR_WIN=$(to_win_path "$PWD")
echo "Base Directory: $BASE_DIR_WIN"

# ==========================================
# メイン処理: try253 から try280 まで
# ==========================================

for i in {292..312}; do
    dir_path="try${i}"
    
    # ディレクトリが存在しない場合はスキップ
    if [ ! -d "$dir_path" ]; then
        # echo "Skipping $dir_path (not found)"
        continue
    fi

    echo "------------------------------------------------"
    echo "Processing: $dir_path"

    ABS_DIR_PATH=$(cd "$dir_path" && pwd)
    INPUT_DIR_WIN=$(to_win_path "$ABS_DIR_PATH")
    OUTPUT_FILE_WIN="${INPUT_DIR_WIN}.scdoc"
    
    # ディレクトリ名から安全なファイル名用文字列を生成
    # (try253 -> try253, ./try253 -> try253)
    SAFE_DIR_NAME=$(echo "$dir_path" | sed 's|^\./||' | sed 's|/|_|g')
    CAT_SCRIPT="batch_script_${SAFE_DIR_NAME}.py"
    LOG_FILE_WIN="${INPUT_DIR_WIN}\\process_log.txt"

# --- Python Script 生成 ---
cat << END_OF_SCRIPT > "$CAT_SCRIPT"
# Python Script, API Version = V23
import System.IO
from System.Collections.Generic import List

def write_log(msg):
    try:
        with open(r'${LOG_FILE_WIN}', 'a') as f:
            f.write(msg + '\n')
    except:
        pass

try:
    write_log("1. Processing: " + r"${INPUT_DIR_WIN}")
    
    input_dir = r"${INPUT_DIR_WIN}"
    output_path = r"${OUTPUT_FILE_WIN}"
    
    files = System.IO.Directory.GetFiles(input_dir, "*.step")
    
    if len(files) > 0:
        write_log("2. Importing " + str(len(files)) + " files...")
        
        DocumentHelper.CreateNewDocument()
        root = GetRootPart()
        
        # 1. インポート
        for f in files:
            DocumentInsert.Execute(f)

        # 2. 【平坦化処理】すべてのボディをルートパーツに移動させる
        write_log("   -> Flattening structure (Moving bodies to Root)...")
        
        # 全階層のボディを取得
        all_bodies = List[IDesignBody]()
        for body in root.GetDescendants[IDesignBody]():
            all_bodies.Add(body)
            
        write_log("   -> Found " + str(all_bodies.Count) + " bodies.")

        if all_bodies.Count > 0:
            # すべてのボディをルート(root)へ移動
            # これによりコンポーネント階層による共有の阻害要因を排除します
            try:
                # ボディを選択状態にする
                sel = Selection.Create(all_bodies)
                # ルートパーツへ移動
                ComponentHelper.MoveBodiesToComponent(sel, root)
            except Exception as e_move:
                write_log("   -> Warning during move: " + str(e_move))

            # 3. 共有トポロジーの実行
            tol_val = ${TOLERANCE_MM}
            write_log("3. Share Topology (Tolerance: " + str(tol_val) + " mm)...")
            
            # オプション設定
            options = ShareTopologyOptions()
            options.Tolerance = MM(tol_val)
            
            # ルートにあるものを全選択（確実に選択されます）
            selection = Selection.SelectAll()
            
            from SpaceClaim.Api.V23 import ShareTopologyType

            root = GetRootPart()
            root.ShareTopology = ShareTopologyType.Share

            # 実行
            ShareTopology.FindAndFix(selection, options)

            write_log("4. Saving...")
            exportOpts = ExportOptions.Create()
            DocumentSave.Execute(output_path, exportOpts)
            
            write_log("5. Success.")
        else:
            write_log("Error: No bodies found.")
    else:
        write_log("Error: No .step files found.")

except Exception as e:
    write_log("FATAL ERROR: " + str(e))

END_OF_SCRIPT
# --- Python Script 終了 ---

    SCRIPT_WIN_PATH=$(to_win_path "$PWD/$CAT_SCRIPT")
    
    echo "  Running SpaceClaim..."
    # SpaceClaimを実行
    "$SC_PATH" /RunScript="$SCRIPT_WIN_PATH" /Headless=False /Splash=False /Welcome=False /ExitAfterScript=True >> batch_process.log 2>&1 < /dev/null

    # 一時スクリプトを削除
    rm "$CAT_SCRIPT"
    echo "  Done."

done

echo "------------------------------------------------"
echo "All jobs completed."
