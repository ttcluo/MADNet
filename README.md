# MADNet
### 📖[**Paper**](https://ieeexplore.ieee.org/document/10387229) | 🖼️[**PDF**](/fig/TTST.pdf)

PyTorch codes for "[Multi-Axis Feature Diversity Enhancement for Remote Sensing Video Super-Resolution](https://ieeexplore.ieee.org/document/10387229)", **IEEE Transactions on Image Processing (IEEE TIP)**, 2025.

- Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Kui Jiang*](https://homepage.hit.edu.cn/jiangkui?lang=zh), [Yuzeng Chen](https://yzcu.github.io/), [Shiqi Wang](https://www.cs.cityu.edu.hk/~shiqwang/), and [Chia-Wen Lin](https://www.ee.nthu.edu.tw/cwlin/)<br>
- Wuhan University, Harbin Institute of Technology,  City University of Hong Kong, and National Tsinghua University

## Abstract
> How to aggregate spatial-temporal information plays an essential role in video super-resolution (VSR) tasks. Despite the remarkable success, existing methods adopt static convolution to encode spatial-temporal information, which lacks flexibility in aggregating information in large-scale remote sensing scenes, as they often contain heterogeneous features (e.g., diverse textures). In this paper, we propose a spatial feature diversity enhancement module (SDE) and channel diversity exploration module (CDE), which explore the diverse representation of different local patterns while aggregating the global response with compact channel-wise embedding representation. Specifically, SDE introduces multiple learnable filters to extract representative spatial variants and encodes them to generate a dynamic kernel for enriched spatial representation. To explore the diversity in the channel dimension, CDE exploits the discrete cosine transform to transform the feature into the frequency domain. This enriches the channel representation while mitigating massive frequency loss caused by pooling operation. Based on SDE and CDE, we further devise a multi-axis feature diversity enhancement (MADE) module to harmonize the spatial, channel, and pixel-wise features for diverse feature fusion. These elaborate strategies form a novel network for satellite VSR, termed MADNet, which achieves favorable performance against state-of-the-art method BasicVSR++ in terms of average PSNR by 0.14 dB on various video satellites, including JiLin-1, Carbonite-2, SkySat-1, and UrtheCast.
## Network  
 ![image](/img/madnet.png)

## 🧩Install
```
git clone https://github.com/XY-boy/MADNet.git
```
## Environment
 * CUDA 11.1
 * pytorch 1.9.1
 * torchvision 0.10.1
 * build [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md)

## Dataset
Please refer to [MSDTGP](https://github.com/XY-boy/MSDTGP) to prepare the satellite video dataset Jilin-189.

## Train
```
python basicsr/train.py -opt options/train/MADNet/train_MADNet_JiLin.yml
```

## Test
```
python basicsr/train.py -opt options/test/MADNet/test_MADNet.yml
```

## 🖼️ Results
### Visual
 ![image](/img/res2.png)

## Acknowledgments
Our MADNet mainly borrows from [IconVSR](https://github.com/ckkelvinchan/BasicVSR-IconVSR), and [FcaNet](https://github.com/cfzd/FcaNet). Thanks for these excellent open-source works!

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: xiao_yi@whu.edu.cn; xy574475@gmail.com

## Citation
If you find our work helpful in your research, please consider citing it. We appreciate your support！😊

```
@ARTICLE{xiao2024ttst,
  author={Xiao, Yi and Yuan, Qiangqiang and Jiang, Kui and Chen, Yuzeng and, Wang Shiqi and Lin, Chia-Wen},
  journal={IEEE Transactions on Image Processing}, 
  title={Multi-Axis Feature Diversity Enhancement for Remote Sensing Video Super-Resolution}, 
  year={2025},
  volume={34},
  number={},
  pages={1--13},
  doi={10.1109/TIP.2025.3547298}
}
```
