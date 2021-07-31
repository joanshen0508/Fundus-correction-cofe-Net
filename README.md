# Modeling and Enhancing Low-Quality Retinal Fundus Images

**Understanding, Modeling, and Correcting Low-quality Retinal Fundus Images for Clinical Observation and Analysis



[Ziyi Shen](https://sites.google.com/site/ziyishenmi/), [Huazhu Fu](https://hzfu.github.io/), [Jianbin Shen](http://iitlab.bit.edu.cn/mcislab/~shenjianbing/), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)

## **Retinal fundus image dataset**:rocket::rocket::rocket:

In this work, we analyze the ophthalmoscope imaging system and model the reliable degradation of major inferior-quality factors, including uneven illumination, blur, and artifacts.

## **Fundus image correction algorithm** :thought_balloon::thought_balloon::thought_balloon:

Based on the proposed realistic degradation model, a clinical-oriented fundus correction network (Cofe-Net) is proposed to suppress the global degradation factors, and simulataneously preserve anatomical retinal structure and pathological characteristics for clinical observation and analysis.
This algorithm is able to effectively corrects low-quality fundus images without losing retinal details, and benefits medical image analysis applications, e.g., retinal vessel segmentatio and optic disc/cup detection.

Here we will release the code of our degradation algorithm and corresponding parameters for low-quality fundus image generation.
You also could refer to your own requirement and simulate specific images by setting your own data. 

The code of our Cofe-Net will also be released soon.

<img src="./img/101_left.png" width = "100" height = "100" align=center /><img src="./img/1060_right.png" width = "100" height = "100" align=center /><img src="./img/101_left.png" width = "100" height = "100" align=center /><img src="./img/1125_left.png" width = "100" height = "100" align=center /> <img src="./img/101_left_fuse.png" width = "100" height = "100" align=center /><img src="./img/106_left_fuse.png" width = "100" height = "100" align=center /><img src="./img/1060_right_fuse.png" width = "100" height = "100" align=center /><img src="./img/1125_left_fuse.png" width = "100" height = "100" align=center />

<img src="./img/10079_right.png" width = "100" height = "100" align=center /><img src="./img/10084_left.png" width = "100" height = "100" align=center /><img src="./img/10089_left.png" width = "100" height = "100" align=center /><img src="./img/10106_left.png" width = "100" height = "100" align=center /> <img src="./img/10079_right_fuse.png" width = "100" height = "100" align=center /><img src="./img/10084_left_fuse.png" width = "100" height = "100" align=center /><img src="./img/10089_left_fuse.png" width = "100" height = "100" align=center /><img src="./img/10106_left_fuse.png" width = "100" height = "100" align=center />

### Here we release the model we implements, which include:

#### encode_decode_vessel_2scale_share_res_atten: 
multi-scale enhancement network, which is embedded a multi-scale vessel segmentation feature(RSA module) and a attention module(LQA module).

1. [download](https://drive.google.com/drive/folders/16f5S6YR5JjvDaO-wak5DYc0qAFuOox8j?usp=sharing) the vessel feature extraction model and enhacement model, add in the main path.
```
# The test images are seletected from KAGGLE dataset, if you want to know more about the dataset, please download from homepage.
2. Please test the image in imgs folder, which includes the low-quality images and masks. 

3. Run test.py

4. The degradation model can be found in ./degradation_code
```


The release code also include the network achitectures, which are applied in ablation study, please refer to the newtworks.

#### encode_decode_res: 
multi-scale enhancent network, constrained by solely content loss.

#### encode_decode_vessel_res: 
multi-scale enhancement newtork, which is embeded another single-scale vessel segmentation feature(RSA module).

#### encode_decode_vessel_2scale_share_res: 
multi-scale enhancement newtork, which is embeded another multi-scale vessel segmentation feature(RSA module).





### Citation 
```
@article{deep_reitna_enhance,
  title={Modeling and Enhancing Low-Quality Retinal Fundus Images},
  author={Shen, Ziyi and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={3},
  pages={996--1006},
  year={2020},
  publisher={IEEE}
}
```
