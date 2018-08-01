# python升级
mac上ython默认环境是2.7，做开发是不够的，需要升级到3.x.我按照的是3.6.4
# 安装tensorflow
mac上安装TensorFlow有4种方式：virtualenv、本地pip、docker、源码安装，具体参考https://www.tensorflow.org/install/install_mac
我采用的是virtualenv<br/>安装步骤为：<br/>
a)启动终端（即 shell）<br/>您将在此 shell 中执行所有后续步骤。<br/>
b)通过发出以下命令安装 pip 和 Virtualenv：
```shell
sudo easy_install pip
pip install --upgrade virtualenv
```
c)创建 Virtualenv 环境：
``` shell
virtualenv --system-site-packages -p python3 /Users/didi/dipper/tensorflow 
```
d)激活 Virtualenv 环境:
``` shell
source /Users/didi/dipper/tensorflow/bin/activate
```
e)确保安装 pip 8.1 或更高版本
``` shell 
easy_install -U pip
```
f)将 TensorFlow 及其所需的所有软件包安装到活动 Virtualenv 环境中:
``` shell
pip3 install --upgrade tensorflow
```
# 验证TensorFlow是否安装正确
激活虚拟环境<br/>
``` shell
source /Users/didi/dipper/tensorflow/bin/activate
```
激活python
``` shell
python
```
在终端输入如下Python代码：
``` python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
如果打印出结果，就表示成功了：
``` python
Hello, TensorFlow!
```
# 在pycharm里配置TensorFlow
1）找到安装TensorFlow的Python在本地的位置，我的位置是：<br/>
/Users/didi/dipper/tensorflow/bin/python<br/>
2)打开pycharm,新建一个project<br/>
3)打开preference面板，添加Python环境

![image](https://github.com/longshengguoji/ML-Learing-Notes/blob/master/blogs/images/pycharm%E9%85%8D%E7%BD%AEpython%E8%B7%AF%E5%BE%84.png)


