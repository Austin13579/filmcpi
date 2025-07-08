# Utilizing data imbalance to enhance compound-protein interaction prediction models (FilmCPI)

## Package
We use the following Python packages for core development. We test the results with `Python=3.8`
```
numpy==1.24.3
pandas==1.3.0
rdkit==2022.9.5
scikit_learn==1.3.2
torch==1.13.1
```

## Data splitting

Before running codes, you need to split the datasets according to your requirements.
```
cd datasets/
python split.py --ds biosnap --rs 0/1/2/3/4 --split random/up/ud
```

`rs` denotes random seed; 'up' denotes unseen protein split, while 'ud' denotes unseen compound split.

For convenience, you can use the `split.sh`
```
./split.sh
```

## Training

```
cd filmcpi/
python train.py --rs 0 --ds biosnap --split ud --model_type film_cp
```

For convenience, you can use the `run.sh`
```
./run.sh
```
If you want to use FilmCPI<sub>pc</sub> or FilmCPI<sub>concat</sub>, you only need to change the codes in `model.py` in `Line 70, 71`

## Citation

If you find our work helpful to your research, please cite:
```
@article{lin2024utilizing,
  title={Utilizing data imbalance to enhance compound-protein interaction prediction models},
  author={Lin, Wei and Fung, Chi Chung Alan},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
