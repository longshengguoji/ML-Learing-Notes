<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# scikit-learn线性回归算法库小结
scikit-learn对于线性回归提供了比较多的类库,这些类库都可以用来做线性回归分析,本文就对这
些类库的使用做一个总结,重点讲述这些线性回归算法库的不同和各自的使用场景.
线性回归的目的是要得到输出向量Y和输入向量X之间的线性关系,求出线性回归系数&theta;, 也就
是Y=X&theta;.其中Y的维度为m*1,X的维度为m*n,&theta;的维度为n*1.m代表样本个数,n代表样
本特征的维度.
为了得到现行回归系数&theta;, 我们需要定义一个损失函数,一个极小化损失函数的优化方法,以及
一个验证算法的方法.损失函数的不同,损失函数的优化方法不同,验证方法的不同,就形成了不同的
线性回归算法.scikit-learn中的线性回归算法库可以从这三点找出各自的不同点.理解了这些不同
点,对不同的算法使用场景也就好理解了.
## 1、LinearReression

$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
