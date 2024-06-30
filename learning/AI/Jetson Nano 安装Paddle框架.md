# 1. 系统环境
- JetPack 4.4
- Ubuntu 18.04

## 1.1 部署Jetson Nano环境
参考官网上面的教程部署，链接如下：
[https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro)
***注意，JetPack版本必须是4.4，可以通过下面的链接选择版本***
[https://developer.nvidia.com/embedded/jetpack-archive](https://developer.nvidia.com/embedded/jetpack-archive)

## 1.2 Ubuntu源替换
有VPN的小伙伴可以直接跳过这一步
由于Ubuntu默认的源是国外的，下载速度很慢，这里我们将Ubuntu源替换成[清华源](https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/)
```
sudo vim /etc/apt/sources.list
```
```
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-proposed main restricted universe multiverse
```
```
sudo apt-get update
```
# 2. 编译前的准备工作
## 2.1 python环境
系统安装完后，默认会有python3.6版本，为了方便python环境的管理，这里使用venv工具管理python环境，首先需要安装`python3-venv`
```
sudo apt-get install python3-venv
python3 -m venv paddle_env -i https://mirror.baidu.com/pypi/simple #在一个你可以记住的文件夹创建  
```
因为JetPack4.4的nano自带相应的cv2模块，不需要再安装，只需链接即可。
```
ln -s /usr/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so paddle_env/lib/python3.6/site-packages/cv2.so  
```
使TensorRT对虚拟环境python可见。如果初次使用，需要配置CUDA环境变量。
```
# 打开环境变量文件，将下面的三个export语句复制到文件的末尾
vim ~/.bashrc 
# 添加CUDA环境变量  
export CUDA_HOME=/usr/local/cuda  
export PATH=$CUDA_HOME/bin:$PATH  
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH  
# 激活环境变量
source ~/.bashrc  
```
测试cuda是否可见
```
nvcc –version
```
如果是如下提示，那么可以继续接下来的步骤。
```
nvcc: NVIDIA (R) Cuda compiler driver  
Copyright (c) 2005-2018 NVIDIA Corporation  
Built on ...  
Cuda compilation tools, release 10.0, Vxxxxx  
```
为了使Jetpack中的TensorRT对虚拟环境的python可见
```
# 打开环境变量文件，将下面的export语句复制到文件的末尾
vim ~/.bashrc 
# 添加PYTHONPATH环境变量  
export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH  
# 激活环境变量
source ~/.bashrc  
```
进入虚拟环境并测试cv2，paddle_env为虚拟python环境的文件夹名。用下方命令进入虚拟环境
```
source paddle_env/bin/activate  
```
安装必要的包
```
sudo apt-get install python3.6-dev liblapack-dev gfortran libfreetype6-dev libpng-dev libjpeg-dev zlib1g-dev patchelf python3-opencv  
# 用pip安装软件包
pip install cython wheel numpy -i https://mirror.baidu.com/pypi/simple
```
测试cv2和tensorrt    
```
#测试cv2和tensorrt  
python   
import cv2  
import tensorrt  
```
以上没有报错方可继续。

## 2.2 TensorRT库设置
更改TensorRT相关库文件（TensorRT头文件所在目录/usr/include/aarch64-linux-gnu/）：
在NvInferRuntime.h中修改protected: ~IOptimizationProfile() noexcept = default为：
```
virtual ~IOptimizationProfile() noexcept = default;
```
整理TensorRT库文件：
因为编译命令通过指定TensorRT_ROOT的方式找到include和lib中的相关文件。因此需要手动在nano的usr/文件夹中找到上述文件并整理成这种形式。个人的Jetpack 4.4 中库文件分别在`/usr/lib/aarch64-linux-gnu` 和 `/usr/include/aarch64-linux-gnu`。由于镜像中TensorRT的库文件的位置分散在两个不同路径，不建议修改cmake文件来找相应库文件。
先新建一个TensorRT目录
```
mkdir TensorRT
```
将下面的文件拷贝到TensorRT目录
```
─include  
│      NvCaffeParser.h  
│      NvInfer.h  
│      NvInferVersion.h  
│      NvInferPlugin.h  
│      NvOnnxConfig.h  
│      NvOnnxParser.h  
│      NvUffParser.h  
│      NvUtils.h  
│  
└─lib  
        libnvcaffe_parser.a  
        libnvcaffe_parser.so  
        libnvcaffe_parser.so.4  
        libnvcaffe_parser.so.4.1.0  
        libnvinfer.a  
        libnvinfer.so  
        libnvinfer.so.4  
        libnvinfer.so.4.1.0  
        libnvinfer_plugin.a  
        libnvinfer_plugin.so  
        libnvinfer_plugin.so.4  
        libnvinfer_plugin.so.4.1.0  
        libnvparsers.a  
        libnvparsers.so  
        libnvparsers.so.4  
        libnvparsers.so.4.1.0  
```

# 3. 编译Paddle
开启硬件性能模式
```
sudo nvpmodel -m 0 && sudo jetson_clocks
```
如果硬件为Nano，增加swap空间
```
#增加DDR可用空间，Xavier默认内存为16G，所以内存足够，如想在Nano上尝试，请执行如下操作。
sudo fallocate -l 5G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'

#最大的文件打开数量  
ulimit -n 2048   
```

## 3.1 编译NCCL
安装NCCL依赖，此过程可能持续近1h。
```
git clone https://github.com/NVIDIA/nccl.git  
cd nccl
make -j4 src.build
sudo make install  
```
## 3.2 Paddle下载和编译
克隆Paddle的Github
```
# Github
git clone https://github.com/paddlepaddle/paddle
# Gitee
git clone https://gitee.com/paddlepaddle/Paddle
```
查看可用的版本：
```
git tag
```
切换版本：我使用的是2.0.0，不是稳定的版本。不选择1.8.x版本的原因是在尝试中发现该系列版本都会出现报缺少cpuid.h文件的错误，该问题已经在Paddle最新版本修复。
```
git checkout v2.0.0-alpha0
```

编译Paddle

由于Paddle编译会下载很多依赖库，其中有个依赖库比较大，没有VPN很容易下载失败导致编译失败，在编译前可以将cmake/external/protobuf.cmake文件中的https://github.com/protocolbuffers/protobuf.git替换为https://gitee.com/lyl120117/protobuf.git
将cmake/external/openblas.cmake文件中的https://github.com/xianyi/OpenBLAS.git替换为https://gitee.com/lyl120117/OpenBLAS.git

修改cudnn版本号读取文件
```
diff --git a/cmake/cudnn.cmake b/cmake/cudnn.cmake
index 98466d44fc..85d7d66cbb 100644
--- a/cmake/cudnn.cmake
+++ b/cmake/cudnn.cmake
@@ -61,8 +61,9 @@ else()
 endif()
 
 if(CUDNN_FOUND)
-    file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
+    file(READ ${CUDNN_INCLUDE_DIR}/cudnn_version.h CUDNN_VERSION_FILE_CONTENTS)
```
Cmake设置：
之前的准备步骤中我们已经将Paddle编译需要的TensorRT库整理成了适应cmake文件的形式，之后只需要指定TensorRT库的位置即可。接下来的cmake命令指定了编译结果所支持的功能。其他配置可以参考Paddle Inference文档中源码编译的部分。
-DWITH_PYTHON=ON使编译结果内嵌python解释器并编译whl，
-DTENSORRT_ROOT指定TensorRT库的路径，
-DCUDA_ARCH_NAME=Auto指定编译结果只适应当前的GPU架构，可以节省编译时间。
```
mkdir build  
cd build/  
cmake .. -DWITH_CONTRIB=OFF -DWITH_MKL=OFF -DWITH_MKLDNN=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release  -DON_INFER=ON -DWITH_PYTHON=ON -DWITH_XBYAK=OFF -DWITH_NV_JETSON=ON -DPY_VERSION=3.6 -DTENSORRT_ROOT=/home/ps/TensorRT -DCUDA_ARCH_NAME=Auto -DPY_PIP=/home/ps/paddle_env/bin/pip -DPY_WHEEL=/home/ps/paddle_env/bin/wheel
```
开始编译前需要安装python环境
```
sudo apt-get install python3-dev python3-pip python3-venv python3-wheel python3-setuptools
pip install -U pip setuptools -i https://mirror.baidu.com/pypi/simple
```
开始编译（过程很漫长）:
```
＃使用全部核心
make -j4  
# 生成预测lib
make inference_lib_dist   

#安装编译好的paddlepaddle-gpu的whl  
pip install -U python/dist/*.whl #还是在build文件夹  
```



