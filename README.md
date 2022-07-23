# cpystal

結晶学的情報や物性測定データ解析を扱うPythonライブラリ．

# Version
* 0.0.x：α版
* 0.1.x：β版
* 1.x〜：実用版

# Contents
* cpystal
    * core
        * Crystal
        * PPMSResistivity
        * MPMS
        * Energy
        * PhysicalConstant
        * ingredient_flake_dp
    * analysis
        * analysis 
            * compare_powder_Xray_experiment_with_calculation
            * compare_powder_Xray_experiment_with_calculation_of_some_materials
            * make_powder_Xray_diffraction_pattern_in_calculation
            * Crystal_instance_from_cif_data
            * atoms_position_from_p1_file
            * make_struct_file
            * cal_Debye_specific_heat
            * cal_thermal_conductivity
            * brillouin
            * paramagnetization_curie
            * fit_paramagnetism
            * demagnetizating_factor_ellipsoid
            * demagnetizating_factor_rectangular_prism
            * RawDataExpander
            * ExpDataExpander
            * ExpDataExpanderSeebeck
            * ReMakeExpFromRaw
            * AATTPMD
        * spacegroup
            * REF
            * MatrixREF
            * SymmetryOperation
            * PhysicalPropertyTensorAnalyzer
            * spacegroup_to_pointgroup
        * spin_model
            * SpinOperator
            * MultiSpinSystemOperator
    * color
        * Color
        * Gradation
        * rgb_to_hsv
        * hsv_to_rgb
        * rgb_to_hls
        * hls_to_rgb
        * rgb_to_yiq
        * yiq_to_rgb
        * view_gradation
    * graph
        * graph_moment_vs_temp
        * graph_moment_vs_field
        * graph_magnetization_vs_temp
        * graph_magnetization_vs_field
        * graph_Bohr_vs_field
        * graph_Bohr_vs_temp
        * graph_susceptibility_vs_temp
        * graph_powder_Xray_intensity_vs_angle
        * ax_transplant
        * graph_furnace_temperature_profile
        * graph_2zone_temperature_profile
        * graph_susceptibility_vs_temp_CurieWeiss
    * measurement
        * SequenceCommandBase
        * Measure
        * WaitForField
        * WaitForTemp
        * SetField
        * SetTemp
        * SetPower
        * ScanField
        * ScanTemp
        * ScanPower
        * sequence_maker

# Requirements
* numpy
* scipy
* matplotlib
* pymatgen
* tk

# Installation
未実装．
```bash
pip install cpystal
```

# Usage
あとで追加

# License
"cpystal" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
