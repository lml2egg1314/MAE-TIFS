# MAE-TIFS
source code for paper "A New Adversarial Embedding Method for Enhancing Image Steganography"
* 首先，运行data/get_stego.sh文件，对载体图像集中的给定载体图像生成对应空域原始隐写方法（WOW,S-UNIWARD,MiPOD,HILL或者CMD-HILL）下的载密图像，嵌入代价和嵌入概率。
* 其次，运行step2.py文件预训练目标隐写分析器。
* 然后，运行step3.py文件获取载体图像以及多载密图像概率图。
* 最后，运行generate_stego/test_mae_same.m文件生成对抗载密图像。
