# 多結晶体モデルの生成と解析
## 目的
- 事前に用意した多結晶体表面を表す目標画像に整合するような多結晶体モデルを作成し，それに解析条件を付与し，解析まで行うプログラム
## 各ファイルの説明
### CMAES最適化関連
- create_initial_population.py：３次元空間上に初期個体（複数のボロノイ点の３次元座標の組）を生成する
- generate_tess_png.py：各個体のテキストファイルからNeperによるボロノイ分割を実行してtessファイルとpngファイルを生成する
- objective_func.py：目的関数（CMA-ES）について書いたファイル
- generate_next_generation_cmaes.py：適合度を計算し，その値に基づいて次個体を生成する
- run_cmaes_algorithm.py：上の４つのファイルを，指定された条件に達するまで順に繰り返し，目標画像に近い構造を作り出す実行ファイル．
- figure1_2_edge.png：最適化における目標画像
### モーフィング関連
- create_txt_to_obj.py：各個体ファイル(txtファイル)からobjファイルを作成し，good_results内に格納する
- change_tess_values.py：good_results内のtessファイルの値をbenchmark/half_of_half.tessの値に合うようにモーフィング処理を実行し，新たに作成したディレクトリtess_files_replaced内に保存する
- change_obj_values.py：good_results内のobjファイルの値をbenchmark/half_of_half.objの値に合うようにモーフィング処理を行い，新たに作成したディレクトリobj_files_replaced内に保存
- change_tess_to_png.py：tess_files_replaced内のtessファイルを元にNeperを用いて全体画像と上面画像を作成し，それぞれ新たに作成したディレクトリpng_files_replaced, top_png_files_replaced内に保存する
- benchmark：目標画像のobj,全体画像,tess,上面画像ファイルを格納したディレクトリ
- convert_multi_to_single.py：単一OBJファイルを粒ごとの複数の個別OBJファイルに分割する
## 実行手順
※参照ファイルのパスとかはローカルでの設定のままなので実行するときは要変更！  
※実行には各種ライブラリ等のインストールが必要
- CMA-ESによる最適化：
  - 以下のコマンドにより，create_initial_population.py, generate_tess_png.py, objective_func.py, generate_next_generation.pyが条件に達するまで繰り返し実行される．

    ```python3 run_cmaes_algorithm.py```
- モーフィング処理：
  - 最適化の結果作成できたモデルをMORPHINGディレクトリ内にgood_resultsというディレクトリ名で保存してから実行する
  - まずは，create_txt_to_obj.pyを実行し，次にchange_tess_values.pyあるいはchange_obj_values.pyを実行する．
  - それらが終了した後，必要な場合はchange_tess_to_png.pyを実行する．
  - 最後にconvert_multi_to_single.pyを実行して粒ごとに分割する.
