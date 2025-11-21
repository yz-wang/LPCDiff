# Real-Scene Image Dehazing via Laplacian Pyramid-Based Conditional Diffusion Model
Yongzhen Wang, Jie Sun, Heng Liu, Xiao-Ping Zhang and Mingqiang Wei
<br>

## Dependencies
```
pip install -r requirements.txt
````

## How to train?
```
python train.py  
```
Then load the pretrained weights and

```
python train_refine.py  
```

## How to test?
```
python evaluate.py
python evaluate_refine.py
```

## Acknowledgement
Part of the code is adapted from previous works: [Diffusion-Low-Light](https://github.com/JianghaiSCU/Diffusion-Low-Light),. We thank all the authors for their contributions.
