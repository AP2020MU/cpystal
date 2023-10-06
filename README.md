# cpystal

結晶学的情報や物性測定データ解析を扱うPythonライブラリ．

# Version
* v0.2.0

# Recent Updates
* v0.1.0 2023/05/22
    * ディレクトリ構成の大幅な変更
    * `cpystal.mathematics.algebra.core`の球面調和関数関係や`CubicHarmonics`の追加
    * `cpystal.mathematics.algebra.operator`の追加

* v0.2.0 2023/10/06
    * pytestに対応
    * `cpystal.analysis.spacegroup`の拡充
    * `cpystal.mathematics.optimize`の追加
    * sphinxによるドキュメントの作成

# Contents
* cpystal
    * core
        * `Crystal`: インスタンスは1つの単結晶試料に対応
        * `PPMSResistivity`: PPMSでの抵抗測定データ展開
        * `MPMS`: MPMSでの磁化測定データの展開
        * `Energy`: エネルギーの単位変換
        * `PhysicalConstant`: 物理定数
        * `ingredient_flake_dp`: 欠片状物質の秤量に用いる
    * analysis
        * analysis 
            * `cal_Debye_specific_heat`: Debye比熱の計算
            * `cal_thermal_conductivity`: 熱伝導率の計算
            * `brillouin`: Brillouin関数
            * `paramagnetization_curie`: Curie常磁性の計算
            * `fit_paramagnetism`: 常磁性の磁化の磁場依存性のフィッティング(実用的でない)
            * `demagnetizating_factor_ellipsoid`: 回転楕円体の反磁場係数の計算
            * `demagnetizating_factor_rectangular_prism`: 直方体の反磁場係数の計算
        * spacegroup
            * `Atom`: 原子のクラス
            * `SpaceGroupSymmetryOperation`: 空間群に属する対称操作のクラス
            * `CrystalStructure`: 結晶構造を表すクラス
            * `SpinStructure`: 結晶構造上のスピン構造を表すクラス
            * `SymmetryOperation`: 3次元空間における対称操作(の表現行列)
            * `PhysicalPropertyTensorAnalyzer`: 物性テンソルの非ゼロ要素や等価な要素の解析
            * `rotx, roty, rotz`: 3次元空間上の回転行列
            * `rodrigues`: Rodriguesの回転公式
            * `mirror`: 鏡映行列
            * `unit_vector`: 極座標で指定する単位ベクトル
            * `circular_mean`: 角度のベクトル平均
            * `circular_diff`: 角度のベクトル差
            * `circular_interpolate`: 角度の線形補間
            * `spacegroup_to_pointgroup`: 空間群から点群への変換
            * `pointgroup_to_laueclass`: 点群からラウエクラスへの変換
            * `pointgroup_to_crystal_system`: 点群から結晶系への変換
            * `spacegroup_to_bravais_lattice`: 空間群からブラベー格子への変換
            * `is_polar_point_group`: polarな点群であるかどうか
            * `is_polar_space_group`: polarな空間群であるかどうか
            * `is_chiral_point_group`: chiralな点群であるかどうか
            * `is_chiral_space_group`: chiralな空間群であるかどうか
            * `is_achiral_space_group`: achiralな空間群であるかどうか
            * `is_centrosymmetric_point_group`: 中心対称な点群であるかどうか
            * `is_enantiomorphic_space_group`: enantiomorphicな空間群であるかどうか
            * `is_bieberbach_space_group`: bieberbachな空間群であるかどうか
            * `is_sohncke_space_group`: Sohncke空間群であるかどうか
            * `is_symmorphic_space_group`: symmorphicな空間群であるかどうか
            * `crystal_system_to_symbol`: 結晶系から一文字のシンボルへの変換
            * `generate_point_group`: 群の生成元から点群を生成
            * `crystal_system_to_symmetry_directions`: 各結晶系で定義されている対称方向
            * `spacegroup_to_symmetry_directions`: その空間群の対称方向
        * spin_model
            * `SpinOperator`: スピン演算子
            * `MultiSpinSystemOperator`: 多スピン系の演算子
        * thermal_conduction
            * `RawDataExpander`: 熱伝導測定における生データの展開(prefixが'Raw'のファイル)
            * `ExpDataExpander`: 熱伝導特性測定における実験データの展開(prefixが'Exp'のファイル)
            * `ExpDataExpanderSeebeck`: 熱電測定における実験データの展開(prefixが'Exp'のファイル)
            * `ReMakeExpFromRaw`: 生データから実験データを再計算
            * `AATTPMD`: 生データを可視化
        * xrd
            * `atoms_position_from_p1_file`: p1ファイルから原子座標を取得
            * `make_struct_file`: structファイルを生成
            * `compare_powder_Xray_experiment_with_calculation`: 粉末X線回折実験データと理論計算の比較
            * `compare_powder_Xray_experiment_with_calculation_of_some_materials`: 粉末X線回折実験データといくつかの物質に対する理論計算の比較
            * `make_powder_Xray_diffraction_pattern_in_calculation`: 粉末X線回折の理論計算
    * color
        * `Color`: 色をRGB,HSV,HLS,YIQ,XYZ,L*a*b*表色系で表現
        * `Gradation`: グラデーションを表現
        * `rgb_to_hsv`: RGB→HSVの変換
        * `hsv_to_rgb`: HSV→RGBの変換
        * `rgb_to_hls`: RGB→HLSの変換
        * `hls_to_rgb`: HLS→RGBの変換
        * `rgb_to_yiq`: RGB→YIQの変換
        * `yiq_to_rgb`: YIQ→RGBの変換
        * `view_gradation`: グラデーションの確認
    * figure
        * `ImageProcessing`: 画像処理クラス
        * `combine_img_list`: 1次元的に画像を連結
        * `combine_img_matrix`: 2次元的に画像を連結
        * `make_gif`: gif画像の作成
        * `get_all_img`: カレントディレクトリ内のすべての画像を取得
        * `rename_ordered`: カレントディレクトリ内の全画像を取得して名前の順序を保存したまま連番に書き換え
    * graph
        * `graph_moment_vs_temp`: 磁気モーメントの温度依存性のグラフを簡易的に描画
        * `graph_moment_vs_field`: 磁気モーメントの磁場依存性のグラフを簡易的に描画
        * `graph_magnetization_vs_temp`: 磁化の温度依存性のグラフを簡易的に描画
        * `graph_magnetization_vs_field`: 磁化の磁場依存性のグラフを簡易的に描画
        * `graph_Bohr_vs_temp`: 磁化(Bohr磁子単位)の温度依存性のグラフを簡易的に描画
        * `graph_Bohr_vs_field`: 磁化(Bohr磁子単位)の磁場依存性のグラフを簡易的に描画
        * `graph_susceptibility_vs_temp`: 磁化率の温度依存性のグラフを簡易的に描画
        * `graph_susceptibility_vs_temp_CurieWeiss`: 磁化率とその逆数の温度依存性のグラフを簡易的に描画
        * `graph_powder_Xray_intensity_vs_angle`: XRD回折強度の角度依存性のグラフを簡易的に描画
        * `ax_transplant`: matplotlibのaxesオブジェクトから情報を抽出して移植
        * `graph_furnace_temperature_profile`: 電気炉の温度プロファイルの描画
        * `graph_2zone_temperature_profile`: 2ゾーン炉の温度プロファイルの描画
    * mathematics
        * algebra
            * core
                * `REF`: 有理拡大体$\mathbb{Q}(\sqrt{p})$
                * `MatrixREF`: 有理拡大体$\mathbb{Q}(\sqrt{p})$上の行列
                * `PolyInt`: 整数係数多項式
                * `CubicHarmonics`: 球テンソル関数
            * operator
                * `JOperator`: 一般角運動量演算子
                * `StevensJOperator`: Stevens演算子
        * linalg
            * `triu_inv`: 上三角行列の逆行列
            * `tril_inv`: 下三角行列の逆行列
            * `Jacobi`: Jocobi法による斉次線型方程式の解
            * `GaussSeidel`: Gauss-Seidel法による斉次線型方程式の解
            * `TDMA`: 三重対角行列を係数行列とする斉次線型方程式の解
    * measurement
        * `SequenceCommandBase`: シークエンスコマンドの基底クラス
        * `Measure`: 「測定」を指示するシークエンスコマンド
        * `WaitForField`: 「磁場安定待ち」を指示するシークエンスコマンド
        * `WaitForTemp`: 「温度安定待ち」を指示するシークエンスコマンド
        * `SetField`: 「磁場値の設定」を指示するシークエンスコマンド
        * `SetTemp`: 「温度値の設定」を指示するシークエンスコマンド
        * `SetPower`: 「ヒーター出力値の設定」を指示するシークエンスコマンド
        * `ScanField`: 「磁場スキャン」を指示するシークエンスコマンド
        * `ScanTemp`: 「温度スキャン」を指示するシークエンスコマンド
        * `ScanPower`: 「ヒーター出力スキャン」を指示するシークエンスコマンド
        * `sequence_maker`: シークエンスを生成してcsvファイルを出力　

# Requirements
* matplotlib
* more_itertools
* numpy
* pymatgen
* scipy
* sympy
* tk
* tqdm

# Installation
未実装．
```bash
pip install cpystal
```

# Usage
あとで追加

# License
"cpystal" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
