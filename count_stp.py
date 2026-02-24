import os
import re

def main():
    output_file = 'result.txt'
    results = []
    
    # 現在のディレクトリ内を走査
    for entry in os.listdir('.'):
        # ディレクトリであり、名前が 'try' で始まり、その後に数字が続くかチェック
        if os.path.isdir(entry):
            match = re.match(r'^try(\d+)$', entry)
            if match:
                # 数字部分を取得（ソート用に整数化）
                dir_num = int(match.group(1))
                
                # ディレクトリ内の.stpファイルをカウント
                stp_count = 0
                try:
                    files = os.listdir(entry)
                    stp_count = len([f for f in files if f.endswith('.step')])
                except OSError:
                    print(f"Error reading {entry}")
                    continue

                results.append((dir_num, stp_count))

    # ディレクトリ番号順(1, 2, 10...)にソート
    results.sort(key=lambda x: x[0])

    # 結果をファイルに出力
    with open(output_file, 'w') as f:
        # ヘッダー（必要なければ削除してください）
        f.write("Dir_Number STP_Count\n")
        for num, count in results:
            f.write(f"{num} {count}\n")

    print(f"集計完了: {output_file} に保存しました。")

if __name__ == '__main__':
    main()
