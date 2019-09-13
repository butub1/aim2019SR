# aim2019SR

## Requirements

* python 3.x
* pytorch 1.1 

## inference
 change the data_dir and output_dir in `eval.py/main()` function with your own dirs.
```shell
python3 eval.py
```

## test script

```shell
python3 test_velocity --model_dir <your model_dir>
```
## model and pruned channel config
 - model_41.pt
 - prune.txt
## model performence
 - PSNR 28.89
 - param 818432
 - time 0.083 (test for 0001x4.png)
