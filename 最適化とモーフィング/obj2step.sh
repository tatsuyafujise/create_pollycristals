#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 設定: 処理するtryディレクトリの範囲
# try1 ～ try89
# ==========================================
START_NUM=31
END_NUM=89

# ===== ディレクトリ設定 (master_loop.shと同一構成) =====
BASE_DIR="$(pwd)"
PARENT_DIR="$(dirname "${BASE_DIR}")"
MORPH_DIR="${PARENT_DIR}/morphing"
TRY_BASE_DIR="${MORPH_DIR}/try"

# 【重要】obj2stepの「原本」がある場所
MASTER_OBJ_TOOLS_DIR="${MORPH_DIR}/try_obj"

# ツール類の事前チェック
if [ ! -d "${MASTER_OBJ_TOOLS_DIR}" ]; then
    echo "[ERROR] Tool directory ${MASTER_OBJ_TOOLS_DIR} does not exist."
    exit 1
fi

if [ ! -d "${MASTER_OBJ_TOOLS_DIR}/squashfs-root" ]; then
    echo "[ERROR] squashfs-root not found in ${MASTER_OBJ_TOOLS_DIR}."
    exit 1
fi

echo "=========================================="
echo "Start Batch Step 4 (OBJ to STEP)"
echo "Range: try${START_NUM} -> try${END_NUM}"
echo "=========================================="

# ===== メインループ (1～89) =====
for (( i=START_NUM; i<=END_NUM; i++ )); do
    
    TARGET_DIR_NAME="try${i}"
    NEXT_TRY_DIR="${TRY_BASE_DIR}/${TARGET_DIR_NAME}"
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    HOST_NAME="$(hostname)"

    echo ""
    echo "------------------------------------------"
    echo "Processing: ${TARGET_DIR_NAME} (${i}/${END_NUM})"
    echo "Path: ${NEXT_TRY_DIR}"
    echo "------------------------------------------"

    # ディレクトリ存在確認
    if [ ! -d "${NEXT_TRY_DIR}" ]; then
        echo "[SKIP] Directory not found: ${NEXT_TRY_DIR}"
        continue
    fi

    # 変換対象のOBJを探す
    SEARCH_ROOT="${NEXT_TRY_DIR}/obj_files_replaced"
    
    if [ ! -d "${SEARCH_ROOT}" ]; then
        echo "[SKIP] 'obj_files_replaced' not found in ${TARGET_DIR_NAME}"
        continue
    fi

    # 最新のサブディレクトリを取得
    LATEST_SUBDIR=$(ls -1 "${SEARCH_ROOT}" | sort -r | head -n 1)
    
    if [ -z "${LATEST_SUBDIR}" ]; then
        echo "[SKIP] No subdirectories in ${SEARCH_ROOT}"
        continue
    fi

    FULL_SEARCH_PATH="${SEARCH_ROOT}/${LATEST_SUBDIR}"
    TARGET_OBJ=$(ls -1 "${FULL_SEARCH_PATH}"/*.obj 2>/dev/null | sort | head -n 1)

    if [ ! -f "${TARGET_OBJ}" ]; then
        echo "[SKIP] No .obj file found in ${FULL_SEARCH_PATH}"
        continue
    fi

    echo "  -> Found Target: ${TARGET_OBJ}"
    
    # === 隔離環境の構築 ===
    # 共有の try_obj は使わず、一時フォルダを作る
    ISO_OBJ_WORK_DIR="${MORPH_DIR}/obj_work_${HOST_NAME}_try${i}_${RUN_ID}"
    mkdir -p "${ISO_OBJ_WORK_DIR}"
    
    # 必要なスクリプトをコピー
    cp "${MASTER_OBJ_TOOLS_DIR}/obj2step.sh" "${ISO_OBJ_WORK_DIR}/"
    cp "${MASTER_OBJ_TOOLS_DIR}/split_by_group.py" "${ISO_OBJ_WORK_DIR}/"
    cp "${MASTER_OBJ_TOOLS_DIR}/obj2solid_step.py" "${ISO_OBJ_WORK_DIR}/"
    
    # squashfs-root はシンボリックリンク (高速化)
    ln -s "${MASTER_OBJ_TOOLS_DIR}/squashfs-root" "${ISO_OBJ_WORK_DIR}/squashfs-root"
    
    # 処理対象のOBJを input.obj としてコピー
    cp "${TARGET_OBJ}" "${ISO_OBJ_WORK_DIR}/input.obj"
    
    # === 変換実行 ===
    pushd "${ISO_OBJ_WORK_DIR}" > /dev/null
    
    echo "  -> Running conversion..."
    if bash obj2step.sh > convert.log 2>&1; then
        echo "  -> [SUCCESS] Conversion finished."
        
        # 結果(stepsフォルダ)を try〇 内に移動
        if [ -d "steps" ]; then
            DEST_PATH="${NEXT_TRY_DIR}/steps_result"
            
            # 既にフォルダがある場合は削除して入れ替え
            if [ -d "${DEST_PATH}" ]; then
                rm -rf "${DEST_PATH}"
            fi
            
            mv "steps" "${DEST_PATH}"
            echo "  -> Saved to: ${DEST_PATH}"
        fi
    else
        echo "  -> [FAILURE] obj2step.sh failed."
        # エラーログを保存
        mv convert.log "${NEXT_TRY_DIR}/obj2step_fail_${RUN_ID}.log"
        echo "  -> Log saved to ${NEXT_TRY_DIR}/obj2step_fail_${RUN_ID}.log"
    fi
    
    popd > /dev/null
    
    # === 後始末 ===
    rm -rf "${ISO_OBJ_WORK_DIR}"
    echo "  -> Cleanup temporary workspace."

done

echo ""
echo "=========================================="
echo "Batch processing completed."
echo "=========================================="
