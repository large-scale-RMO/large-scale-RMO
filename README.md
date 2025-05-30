# 子空间自适应的黎曼元优化方法（RSAMO）

##实验1：优化正交神经网络用于图像分类实验
### 📁 项目目录结构

orthogonal\_DNN\_optimization/

├── vgg16/          # 针对 VGG16 的实验

├── resnet18/       # 针对 ResNet18 的实验

├── resnet50/       # 针对 ResNet50 的实验


每个子目录下包含多种优化器的实现版本，如 RSAMO、RMO、传统黎曼优化器，以及正交正则化方法等。

默认数据集为 CIFAR-10，用户可通过修改train.py中的
’root='/home/smbu/mcislab/ypl/rsamo/cifar10/data‘，按需替换为 CIFAR-100、SVHN 等。



---

### 🚀 实验运行方式


#### 1. RSAMO（子空间自适应黎曼元优化方法）

- 路径：
./<模型目录>/rsamo/

- 运行指令：
python train.py


#### 2. RMO（Riemannian Meta-Optimization）

* 路径：  ./<模型目录>/RMO/

* 运行指令：python train.py

#### 3. I-RMO（基于隐式微分的 RMO）

* 路径：  ./<模型目录>/I_RMO/

* 运行指令：
  python train.py


#### 4. 传统黎曼优化器

包括以下几种方法：

* RSGD

* RSGDM

* RSVRG

* RMSProp

* RASA

* 路径示例：

  ```
  ./<模型目录>/RSGD/
  ./<模型目录>/RSGDM/
  ./<模型目录>/RSVRG/
  ./<模型目录>/RMSProp/
  ./<模型目录>/RASA/
  ```

* 统一运行指令：
  python train.py


备注：使用传统优化方法优化ResNet50网络的优化器代码与vgg16\resnet18下的相同，可直接复制，因此不再重复赘述

#### 5. 正交正则化方法

包含方法：

* SO

* DSO

* SRIP

* SRIP\_P

* MC

* 路径：  ./<模型目录>/soft/


* 运行指令：
  python train.py
 

* 更换正则化方法：修改 `train.py` 第 442 行的正则项函数。例如：oloss = SRIP(param)

---

## 实验2：消融实验

* 路径：  ./vgg16/rsamo/

* 在 `train.py` 中第 15 行修改 `meta_optimizer` 导入项，以控制子空间适应策略：

仅行子空间适应     ——> `from optimizer.meta_optimizer import meta_optimizer_L`    

仅列子空间适应     ——> `from optimizer.meta_optimizer import meta_optimizer_R`    

欧式 LSTM 优化器  ——> `from optimizer.meta_optimizer import meta_optimizer_LSTM` 


* 训练命令：
  python train.py
 

---

## 实验3：YaleB 数据集人脸识别实验

* 路径：
  ./Stiefel_face/
  

* 训练命令：
  python train.py
 

---

## 实验4：MNIST 数据集上的 PCA 实验

* 路径：  ./PCA/
* 下载数据集并放置于./PCA/data/：https://drive.google.com/drive/folders/1FMN8SIrWSC8MuCAKJS0fLlrH3hojKNx0?usp=drive_link

* 训练命令：  python train.py



