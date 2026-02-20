# 多結晶体モデルの生成と解析
## 目的
- 事前に用意した多結晶体表面を表す目標画像に整合するような多結晶体モデルを作成し，それに解析条件を付与し，解析まで行うプログラム
## 各ファイルの説明
- create_initial_population.py：３次元空間上に初期個体（複数のボロノイ点の３次元座標の組）を生成するファイル
- generate_tess_png.py：各個体のテキストファイルからNeperによるボロノイ分割を実行してtessファイルとpngファイルを生成するファイル
- objective_func.py：目的関数（CMA-ES）について書いたファイル
- generate_next_generation_cmaes.py：適合度を計算し，その値に基づいて次個体を生成するファイル
- run_cmaes_algorithm.py：上の４つのファイルを，指定された条件に達するまで順に繰り返し，目標画像に近い構造を作り出す実行ファイル．
- figure1_2_edge.png：最適化における目標画像

## 実行手順
※参照ファイルのパスとかはローカルでの設定のままなので実行するときは要変更！
※実行には各種ライブラリ等のインストールが必要
- CMA-ESによる最適化：
  - 以下のコマンドにより，create_initial_population.py, generate_tess_png.py, objective_func.py, generate_next_generation.pyが条件に達するまで繰り返し実行される．

    ```python3 run_cmaes_algorithm.py```
