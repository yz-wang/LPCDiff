# Real-Scene Image Dehazing via Laplacian Pyramid-Based Conditional Diffusion Model (TMM'2025)
Authors: Yongzhen Wang, Jie Sun, Heng Liu, Xiao-Ping Zhang and Mingqiang Wei

[[Paper Link]](https://ieeexplore.ieee.org/document/11249427)

### Abstract

Recent diffusion models have demonstrated exceptional efficacy across various image restoration tasks, but still suffer from time-consuming and substantial computational resource consumption. To address these challenges, we present LPCDiff, a novel Laplacian Pyramid-based Conditional Diffusion model designed for real-scene image dehazing. LPCDiff leverages the Laplacian pyramid decomposition to decouple the input image into two components: the low-resolution low-pass image and the high-frequency residuals. These components are subsequently reconstructed through a diffusion model and a well-designed high-frequency residual recovery module. With such a strategy, LPCDiff can substantially accelerate inference speed and reduce computational costs without sacrificing image fidelity. In addition, the framework empowers the model to capture intrinsic high-frequency details and low-frequency structural information within the image, resulting in sharper and more realistic haze-free outputs. Moreover, to extract more valuable information from the limited training data, we introduce a low-frequency refinement module to further enhance the intricate details of the final dehazed images. Through extensive experimentation, our method significantly outperforms 12 state-of-the-art approaches on three real-world and one synthetic image dehazing benchmarks. Code is available at https://github.com/yz-wang/LPCDiff.

#### If you find the resource useful, please cite the following :- )

```
@ARTICLE{11249427,
  author={Wang, Yongzhen and Sun, Jie and Liu, Heng and Zhang, Xiao-Ping and Wei, Mingqiang},
  journal={IEEE Transactions on Multimedia}, 
  title={Real-Scene Image Dehazing Via Laplacian Pyramid-Based Conditional Diffusion Model}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2025.3632694}}
```  

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
