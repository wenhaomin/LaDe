
## Run
1. requirements

python version: 3.9
```shell
# install packages
pip install -r requirements.txt
```
2. put the data into /data/raw/
The structure of data/raw/ should be like:
/data/raw/  
├── delivery    
│   ├── delivery_sh.csv   
│   └── ...    
└── pickup  
    ├── pickup_sh.csv  
    └── ...  


3. run route prediction task


3.1 constructing route prediction dataset  
```shell
python data/dataset_pickup.py
```
The constructed dataset will be stored in /data/dataset/. 

3.2 model training  
```shell
python run.py
```


## Citing
If you find this repository helpful, please cite this paper:

```shell
@software{pytorchgithub,
    author = {xx},
    title = {xx},
    url = {xx},
    version = {0.6.x},
    year = {2021},
}
```

# Leaderboard


| Method       | HR@3         | KRC          | LSD         | ED          |
|--------------|--------------|--------------|-------------|-------------|
| TimeGreedy   | 57.65        | 31.81        | 5.54        | 2.15        |
| DistanceGreedy | 60.77        | 39.81        | 5.54        | 2.15        |
| OR-Tools     | 66.21        | 47.60        | 4.40        | 1.81        |
| LightGBM     | 73.76        | 55.71        | 3.01        | 1.84        |
| FDNET        | 73.27 ± 0.47 | 53.80 ± 0.58 | 3.30 ± 0.04 | 1.84 ± 0.01 |
| DeepRoute    | 74.68 ± 0.07 | 56.60 ± 0.16 | 2.98 ± 0.01 | 1.79 ± 0.01 |
| Graph2Route  | 74.84 ± 0.15 | 56.99 ± 0.52 | 2.86 ± 0.02 | 1.77 ± 0.01 |
