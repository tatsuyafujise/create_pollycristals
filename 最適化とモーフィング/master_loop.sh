#!/usr/bin/env bash
set -euo pipefail

# ===== 設定: ループ回数 =====
LOOP_COUNT=10

# ===== ディレクトリ設定 =====
BASE_DIR="$(pwd)"
PARENT_DIR="$(dirname "${BASE_DIR}")"
MORPH_DIR="${PARENT_DIR}/morphing"
TRY_BASE_DIR="${MORPH_DIR}/try"

# 【重要】obj2stepの「原本」がある場所 (ここには書き込まず、参照のみ行う)
MASTER_OBJ_TOOLS_DIR="${MORPH_DIR}/try_obj"

# 重要な依存ファイル（存在確認用）
REF_IMG="figure1_2_edge.png"
REF_TESS="half_of_half.tess"

# 最終的に回収するディレクトリ名
TARGET_DIRS=("good_results" "obj_files_replaced" "png_files_replaced" "tess_files_replaced" "top_png_files_replaced")

# ===== 事前チェック =====
REQUIRED_FILES=("run_cmaes_algorithm.py" "objective_func.py" "${REF_IMG}")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${BASE_DIR}/${f}" ]; then
        echo "[ERROR] ${f} not found in ${BASE_DIR}"
        exit 1
    fi
done

if [ ! -f "${MORPH_DIR}/${REF_TESS}" ]; then
    echo "[ERROR] ${REF_TESS} not found in ${MORPH_DIR}"
    exit 1
fi

# obj2step用の原本ディレクトリ確認
if [ ! -d "${MASTER_OBJ_TOOLS_DIR}" ]; then
    echo "[ERROR] Tool directory ${MASTER_OBJ_TOOLS_DIR} does not exist."
    echo "       Please place obj2step.sh, *.py, and squashfs-root there."
    exit 1
fi
# squashfs-rootの存在確認
if [ ! -d "${MASTER_OBJ_TOOLS_DIR}/squashfs-root" ]; then
    echo "[ERROR] squashfs-root not found in ${MASTER_OBJ_TOOLS_DIR}."
    exit 1
fi

# ===== 作業用ディレクトリの親フォルダ作成 =====
EXEC_WORK_ROOT="${BASE_DIR}/_workspaces"
mkdir -p "${EXEC_WORK_ROOT}"

# ===== メインループ開始 =====
echo "=========================================="
echo "Start Parallel Loop (Final): ${LOOP_COUNT} times"
echo "Host: $(hostname)"
echo "=========================================="

for (( i=1; i<=LOOP_COUNT; i++ ))
do
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    
    echo ""
    echo "##########################################"
    echo "  ITERATION: ${i} / ${LOOP_COUNT}"
    echo "##########################################"
    
    # ---------------------------------------------------------
    # STEP 1: CMA-ES 実行 (完全隔離環境)
    # ---------------------------------------------------------
    
    # 1-1. 自分専用の実行フォルダを作成
    EXEC_DIR_NAME="exec_$(hostname)_$$_${RUN_ID}"
    EXEC_DIR="${EXEC_WORK_ROOT}/${EXEC_DIR_NAME}"
    
    echo "[Step 1] Creating isolated execution workspace: ${EXEC_DIR}"
    mkdir -p "${EXEC_DIR}"
    
    cp "${BASE_DIR}"/*.py "${EXEC_DIR}/"
    cp "${BASE_DIR}/${REF_IMG}" "${EXEC_DIR}/"
    
    cd "${EXEC_DIR}"
    
    GENERATED_DIR_NAME="CMAES_results"
    
    # 保存先パス (共通ストレージ)
    SAVE_ROOT="${BASE_DIR}/result"
    DST_DIR="${SAVE_ROOT}/stage1_${RUN_ID}_iter${i}_$(hostname)"
    LOG_DIR="${DST_DIR}/logs"
    mkdir -p "${LOG_DIR}"

    echo "[Step 1] Running CMA-ES Algorithm..."
    if ! python3 "run_cmaes_algorithm.py" > "${LOG_DIR}/run.log" 2>&1; then
        echo "[ERROR] CMA-ES failed. Skipping to next iteration." >&2
        cd "${BASE_DIR}"
        rm -rf "${EXEC_DIR}"
        continue
    fi

    # 1-4. 結果回収
    if [ ! -d "${GENERATED_DIR_NAME}" ]; then
        echo "[ERROR] CMAES_results not found! Skipping..." >&2
        cd "${BASE_DIR}"
        rm -rf "${EXEC_DIR}"
        continue
    fi

    mv "${GENERATED_DIR_NAME}" "${DST_DIR}/"

    cat > "${DST_DIR}/meta.txt" <<EOF
iter: ${i}
run_id: ${RUN_ID}
host: $(hostname)
date: $(date)
work_dir: ${EXEC_DIR}
EOF

    cd "${BASE_DIR}"
    rm -rf "${EXEC_DIR}"


    # ---------------------------------------------------------
    # STEP 2: Morphing Python Scripts 実行 (完全隔離環境)
    # ---------------------------------------------------------
    
    WORK_DIR_NAME="morph_$(hostname)_$$_${RUN_ID}"
    WORK_DIR="${MORPH_DIR}/${WORK_DIR_NAME}"
    
    echo "[Step 2] Creating isolated morphing workspace: ${WORK_DIR}"
    mkdir -p "${WORK_DIR}"

    cp "${MORPH_DIR}"/*.py "${WORK_DIR}/"
    cp "${MORPH_DIR}/${REF_TESS}" "${WORK_DIR}/"

    MOVED_SRC="${DST_DIR}/${GENERATED_DIR_NAME}"
    MORPH_DST="${WORK_DIR}/good_results"
    
    mkdir -p "${MORPH_DST}"
    echo "[Step 2] Copying results for processing..."
    cp -r "${MOVED_SRC}/." "${MORPH_DST}/"

    cd "${WORK_DIR}"
    
    if python3 "create_txt_to_obj.py" > /dev/null && \
       python3 "change_tess_values.py" > /dev/null && \
       python3 "change_obj_values.py" > /dev/null && \
       python3 "change_tess_to_png.py" > /dev/null; then
       
       echo "[Step 2] Morphing scripts finished successfully."
    else
       echo "[ERROR] Morphing scripts failed. Skipping to next iteration." >&2
       cd "${BASE_DIR}"
       rm -rf "${WORK_DIR}"
       continue
    fi

    # ---------------------------------------------------------
    # STEP 3: アーカイブ (フォルダ番号決定のみ排他制御)
    # ---------------------------------------------------------
    mkdir -p "${TRY_BASE_DIR}"
    LOCK_DIR="${TRY_BASE_DIR}/dir_lock"

    echo "[Step 3] Acquiring lock and Archiving..."
    # ロック取得待ち
    while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
        sleep 0.$(( RANDOM % 5 + 1 ))
    done

    # --- クリティカルセクション開始 ---
    MAX_NUM=$(find "${TRY_BASE_DIR}" -maxdepth 1 -type d -name 'try*' | \
              sed 's/.*try//' | grep -E '^[0-9]+$' | sort -n | tail -1)

    if [ -z "${MAX_NUM}" ]; then
        NEXT_NUM=1
    else
        NEXT_NUM=$((MAX_NUM + 1))
    fi

    NEXT_DIR_NAME="try${NEXT_NUM}"
    NEXT_TRY_DIR="${TRY_BASE_DIR}/${NEXT_DIR_NAME}"
    
    # フォルダ予約
    mkdir -p "${NEXT_TRY_DIR}"
    
    # ロック解除 (フォルダ名さえ決まれば、以降のコピーは競合しない)
    rmdir "${LOCK_DIR}"
    # --- クリティカルセクション終了 ---
    
    echo "[Step 3] Archiving to: ${NEXT_TRY_DIR}"

    for target in "${TARGET_DIRS[@]}"; do
        if [ -d "${target}" ]; then
            mv "${target}" "${NEXT_TRY_DIR}/"
        else
            echo "[WARNING] Directory ${target} not found in workspace."
        fi
    done
    
    echo "computed_by: $(hostname)" > "${NEXT_TRY_DIR}/host_info.txt"

    # Step 2の後始末
    cd "${BASE_DIR}"
    rm -rf "${WORK_DIR}"

    # ---------------------------------------------------------
    # STEP 4: STEPファイル変換 (完全隔離環境で実行)
    # ---------------------------------------------------------
    echo "[Step 4] Converting OBJ to STEP..."

    SEARCH_ROOT="${NEXT_TRY_DIR}/obj_files_replaced"
    
    if [ -d "${SEARCH_ROOT}" ]; then
        LATEST_SUBDIR=$(ls -1 "${SEARCH_ROOT}" | sort -r | head -n 1)
        
        if [ -n "${LATEST_SUBDIR}" ]; then
            FULL_SEARCH_PATH="${SEARCH_ROOT}/${LATEST_SUBDIR}"
            TARGET_OBJ=$(ls -1 "${FULL_SEARCH_PATH}"/*.obj 2>/dev/null | sort | head -n 1)
            
            if [ -f "${TARGET_OBJ}" ]; then
                echo "  -> Found Target: ${TARGET_OBJ}"
                
                # === 隔離環境の構築 ===
                # 共有の try_obj は使わず、一時フォルダを作る
                ISO_OBJ_WORK_DIR="${MORPH_DIR}/obj_work_$(hostname)_$$_${RUN_ID}"
                mkdir -p "${ISO_OBJ_WORK_DIR}"
                
                # 1. 必要なスクリプトをコピー
                cp "${MASTER_OBJ_TOOLS_DIR}/obj2step.sh" "${ISO_OBJ_WORK_DIR}/"
                cp "${MASTER_OBJ_TOOLS_DIR}/split_by_group.py" "${ISO_OBJ_WORK_DIR}/"
                cp "${MASTER_OBJ_TOOLS_DIR}/obj2solid_step.py" "${ISO_OBJ_WORK_DIR}/"
                
                # 2. 重い squashfs-root はコピーせず、シンボリックリンクを貼る (高速化)
                ln -s "${MASTER_OBJ_TOOLS_DIR}/squashfs-root" "${ISO_OBJ_WORK_DIR}/squashfs-root"
                
                # 3. 処理対象のOBJを input.obj としてコピー
                cp "${TARGET_OBJ}" "${ISO_OBJ_WORK_DIR}/input.obj"
                
                # === 実行 ===
                pushd "${ISO_OBJ_WORK_DIR}" > /dev/null
                
                if bash obj2step.sh > convert.log 2>&1; then
                    echo "  -> Conversion Success."
                    
                    # 結果(stepsフォルダ)を try〇 内に移動
                    if [ -d "steps" ]; then
                        mv "steps" "${NEXT_TRY_DIR}/steps_result"
                        echo "  -> Moved results to ${NEXT_TRY_DIR}/steps_result"
                    fi
                else
                    echo "[ERROR] obj2step.sh failed. Check ${ISO_OBJ_WORK_DIR}/convert.log if needed."
                    # デバッグ用にログを保存する場合
                    mv convert.log "${NEXT_TRY_DIR}/obj2step_fail.log"
                fi
                
                popd > /dev/null
                
                # === 後始末 ===
                rm -rf "${ISO_OBJ_WORK_DIR}"
                
            else
                echo "[WARNING] No .obj file found in ${FULL_SEARCH_PATH}"
            fi
        else
            echo "[WARNING] No subdirectories found in ${SEARCH_ROOT}"
        fi
    else
        echo "[WARNING] ${SEARCH_ROOT} does not exist. Skipping conversion."
    fi

    echo "[SUCCESS] Iteration ${i} completed."
    sleep 1

done

echo "Done."
