# xDeepH

--------------------------------------------------------------------------------
The current code is the implementation of the xDeepH (extended Deep Hamiltonian) method described in the paper
[*Deep-learning electronic-structure calculation of magnetic superstructures*](https://arxiv.org/abs/2211.10604) (accepted by Nature Computational Science).

The current xDeepH repository is based on [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3)
and will be integrated into [DeepH-pack](https://github.com/mzjb/DeepH-pack) in the future.

## Requirements

To train the xDeepH model, Python 3.9 interpreter and following packages are required：
- NumPy
- SciPy
- PyTorch = 1.9.1
- PyTorch Geometric = 1.7.2
- e3nn = 0.3.5
- pymatgen
- h5py
- TensorBoard
- pathos
- psutil

In Linux, you can quickly achieve the requirements by running

```bash
# install miniconda with python 3.9
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh

# install packages by conda
conda install numpy
conda install scipy
conda install pytorch==1.9.1 ${pytorch_config}
conda install pytorch-geometric=1.7.2 -c rusty1s -c conda-forge
conda install pymatgen -c conda-forge

# install packages by pip
pip install e3nn==0.3.5
pip install h5py
pip install tensorboard
pip install pathos
pip install psutil
```

with `${pytorch_config}` replaced by your own configuration. You can find how to set it in [the official website of PyTorch](https://pytorch.org/get-started/previous-versions/).


## Demo: xDeepH study on monolayer CrI<sub>3</sub>
The usage of xDeepH is similar to that of DeepH-pack (https://github.com/mzjb/DeepH-pack). 

### Prepare the dataset
Download the processed dataset from [Zenodo](https://zenodo.org/record/7561013/files/monolayer_CrI3.zip?download=1) and unzip it to the directory `data/monolayer_CrI3/`.

Alternatively, you can also use [OpenMX](https://www.openmx-square.org/) to perform
the spin-constrained DFT calculation, and then convert the output of DFT codes to the format
that can be directly read by xDeepH using the `deeph-preprocess` command of DeepH-pack
(https://github.com/mzjb/DeepH-pack/#prepare-the-dataset) to obtain a dataset. You should set
`parse_magnetic_moment` to `True` and `local_coordinate` to `False` in the config file of `deeph-preprocess` 
(https://github.com/mzjb/DeepH-pack/blob/1130ad648d8b02be96e9de0144f8f8cff886a178/deeph/preprocess/preprocess_default.ini#L19).

### Train your model
Based on the downloaded data, you can train an xDeepH model. Run the following command:

```bash
python -u xdeeph-train.py examples/CrI3_monolayer.ini
```

The config `CrI3_monolayer.ini` includes a description of each input variables.

### Model inference
Once you have a trained model, you can use that model to predict the Hamiltonian of some material structures
and get `hamiltonians_pred.h5`.

Like DeepH-pack, you should first calculate the overlap for the desired
materials using `'overlap only' OpenMX` (https://github.com/mzjb/overlap-only-OpenMX), and then convert
it to our format using the `deeph-inference` command of DeepH-pack
(https://github.com/mzjb/DeepH-pack#inference-with-your-model). Please note that the `deeph-inference`
command is only used for converting the overlap format and not for performing model inference.
You should set `task` to `[1]` in the config file of `deeph-inference` 
(https://github.com/mzjb/DeepH-pack/blob/1130ad648d8b02be96e9de0144f8f8cff886a178/deeph/inference/inference_default.ini#L6).

After obtaining the converted overlaps.h5 file, you should create a `magmom.txt` file to specify the magnetic structure of the material you want to predict.
This is a text file where each line corresponds to an atom. The first column indicates whether the atom
is magnetic, with 1 for yes and 0 for no. The second column indicates the magnitude of the magnetic moment.
The third column indicates the polar angle of the magnetic moment in spherical coordinates.
The fourth column indicates the azimuthal angle of the magnetic moment in spherical coordinates,
with angles in degrees. Here is an example of a `magmom.txt` file:
```
1.0e+00  3.46e+00  3.23e+01 -5.62e+01
1.0e+00  3.45e+00  4.39e+01  7.07e+01
0.0e+00  7.75e-02  9.43e+01  1.11e+02
0.0e+00  1.43e-02  1.58e+02  8.73e+01
0.0e+00  4.68e-02  1.32e+02 -3.16e+01
0.0e+00  1.41e-01  1.48e+02  3.06e+01
0.0e+00  7.55e-02  8.40e+01  1.11e+02
0.0e+00  9.92e-03  1.20e+02  8.54e+01
```

Inference of xDeepH can be done by the command:

```bash
python -u xdeeph-eval.py examples/CrI3_monolayer_inference.ini
```

This config also includes a description of each input variables.

### Band structure calculation
The band structure of the material can be calculated from the predicted Hamiltonian.
The calculation is also based on the sparse calculation script from DeepH-pack
(https://github.com/mzjb/DeepH-pack/blob/main/deeph/inference/sparse_calc.jl).
The required Julia environment is also identical to that of DeepH-pack
(https://github.com/mzjb/DeepH-pack#julia).

Run the following command:

```bash
julia sparse_calc.jl -i ${predicted_hamiltonian_path} -o ${output_path} --config config.json
```
where `config.json` is the config for the sparse calculation.
The description of parameters in this config can be found in
https://deeph-pack.deepmodeling.com/en/latest/keyword/inference.html#json-configuration-file.
The predicted `hamiltonians_pred.h5` and files generated by `deeph-inference`
(`overlap.h5`, `info.json`, `rlat.dat`, `orbital_types.dat`, `site_positions.dat`)
must be placed in the `${predicted_hamiltonian_path}` directory. 

