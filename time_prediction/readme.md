
## Run
1. requirements
```shell
# install packages
pip install -r requirements.txt
```
2. put the data into /data/raw/
The structure of data/raw/ should be like:
/data/raw/  
├── delivery    
│   ├── delivery_cq.parquet   
│   └── ...    
└── pickup  
    ├── pickup_cq.parquet  
    └── ...  



3. run time prediction task  

3.1 constructing time prediction dataset  
```shell
python data/dataset_delivery.py
```
The constructed dataset will be stored in /data/dataset/. 

3.2  model training
```shell
python run.py
```



