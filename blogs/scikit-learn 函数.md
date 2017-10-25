# scikit-learn 函数

## 1、sklearn.model_selection.train_test_split 随机划分训练集和测试集

官网文档：http://scikit-learn.org/stable/modules/generated/sklearn.
model_selection.train_test_split.html#sklearn.model_selection.train_test_split<br/>
一般形式：<br/>
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train 
data和testdata，形式为：<br/>
X_train,X_test, y_train, y_test =
cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)<br/>
### 参数解释：<br/>
train_data：所要划分的样本特征集<br/>
train_target：所要划分的样本结果<br/>
test_size：样本占比，如果是整数的话就是样本的数量<br/>
random_state：是随机数的种子。<br/>
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。<br/>
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：<br/>
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

## 2、GridSearchCV
GridSearchCV用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。它存在的意义就是
自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，一旦
数据的量级上去了，很难得出结果。这个时候就是需要动脑筋了。数据量比较大的时候可以使用一个
快速调优的方法——坐标下降。它其实是一种贪心算法：拿当前对模型影响最大的参数调优，直到最优
化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。这个方法的缺点就是可
能会调到局部最优而不是全局最优，但是省时间省力，巨大的优势面前，还是试一试吧，后续可以再
拿bagging再优化。<br/>
GridSearchCV的sklearn官方网址：http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV <br/>
classsklearn.model_selection.GridSearchCV(estimator,param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True,cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise',return_train_score=True)
### 参数解释
estimator：所使用的分类器，如estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10), 并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法。<br/>
param_grid：值为字典或者列表，即需要最优化的参数的取值，param_grid =param_test1，param_test1 = {'n_estimators':range(10,71,10)}。<br/>
scoring :准确度评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。<br/>
cv :交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。<br/>
refit :默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。<br/>
iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。<br/>
verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。<br/>
n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值。<br/>
pre_dispatch：指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次<br/>
### 常用方法和属性
grid.fit()：运行网格搜索<br/>
grid_scores_：给出不同参数情况下的评价结果<br/>
best_params_：描述了已取得最佳结果的参数的组合<br/>
best_score_：成员提供优化过程期间观察到的最好的评分<br/>

## 3、LogisticRegression
逻辑回归，可以做概率预测，也可用于分类，仅能用于线性问题。通过计算真实值与预测值的概率，然后变换成损失函数，求损失函数最小值来计算模型参数，从而得出模型<br/>
官方API：http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html <br/> 
class sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,warm_start=False, n_jobs=1)
### 参数解释
* penalty : str, ‘l1’or ‘l2’, default: ‘l2’，正则化选择参数（惩罚项的种类）<br/>
在调参时如果我们主要的目的只是为了解决过拟合，一般penalty选择L2正则化就够了。但是如果选择L2正则化发现还是过拟合，即预测效果差的时候，就可以考虑L1正则化。另外，如果模型的特征非常多，我们希望一些不重要的特征系数归零，从而让模型系数稀疏化的话，也可以使用L1正则化。<br/>
penalty参数的选择会影响我们损失函数优化算法的选择。即参数solver的选择，如果是L2正则化，那么4种可选的算法{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}都可以选择。但是如果penalty是L1正则化的话，就只能选择‘liblinear’了。这是因为L1正则化的损失函数不是连续可导的，而{‘newton-cg’, ‘lbfgs’,‘sag’}这三种优化算法时都需要损失函数的一阶或者二阶连续导数。而‘liblinear’并没有这个依赖。
* dual : bool, default: False <br/>
Dual只适用于正则化相为l2 liblinear的情况，通常样本数大于特征数的情况下，默认为False。<br/>
* C : float, default: 1.0 <br/>
C为正则化系数λ的倒数，通常默认为1<br/>
* fit_intercept : bool, default: True<br/>
是否存在截距，默认存在<br/>
* intercept_scaling : float, default 1.<br/>
仅在正则化项为"liblinear"，且fit_intercept设置为True时有用<br/>
* solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}, default: ‘liblinear’<br/>
优化算法选择参数<br/>
solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是：<br/>
a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。<br/>
b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。<br/>
c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。<br/>
d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。<br/>
newton-cg, lbfgs和sag这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear通吃L1正则化和L2正则化。<br/>
sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，比如大于10万，sag是第一选择。但是sag不能用于L1正则化，所以当你有大量的样本，又需要L1正则化的话就要自己做取舍了。要么通过对样本采样来降低样本量，要么回到L2正则化。<br/>
**几种优化算法适用情况:**<br/>
a)**liblinear**:适用于小数据集；如果选择L2正则化发现还是过拟合，即预测效果差的时候，就可以考虑L1正则化；如果模型的特征非常多，希望一些不重要的特征系数归零，从而让模型系数稀疏化的话，也可以使用L1正则化。<br/>
b)**libniear**:只支持多元逻辑回归的OvR，不支持MvM，但MVM相对精确。<br/>
c)**lbfgs/newton-cg/sag**:较大数据集，支持one-vs-rest(OvR)和many-vs-many(MvM)两种多元逻辑回归<br/>
d)**sag**:如果样本量非常大，比如大于10万，sag是第一选择；但不能用于L1正则化
* multi_class : str, {‘ovr’, ‘multinomial’}, default:‘ovr’.分类方式选择参数<br/>
ovr即前面提到的one-vs-rest(OvR)，而multinomial即前面提到的many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上<br/>
**OvR和MvM有什么不同:**<br/>
OvR的思想很简单，无论你是多少元逻辑回归，我们都可以看做二元逻辑回归。具体做法是，对于第K类的分类决策，我们把所有第K类的样本作为正例，除了第K类样本以外的所有样本都作为负例，然后在上面做二元逻辑回归，得到第K类的分类模型。其他类的分类模型获得以此类推。<br/>
而MvM则相对复杂，这里举MvM的特例one-vs-one(OvO)作讲解。如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类。<br/>
可以看出OvR相对简单，但分类效果相对略差（这里指大多数样本分布情况，某些样本分布下OvR可能更好）。而MvM分类相对精确，但是分类速度没有OvR快。如果选择了ovr，则4种损失函数的优化方法liblinear，newton-cg,lbfgs和sag都可以选择。但是如果选择了multinomial,则只能选择newton-cg, lbfgs和sag了。<br/>
* class_weight : dictor ‘balanced’, default: None.类型权重参数：（考虑误分类代价敏感、分类类型不平衡的问题）
class_weight参数用于标示分类模型中各种类型的权重，可以不输入，即不考虑权重，或者说所有类型的权重一样。如果选择输入的话，可以选择balanced让类库自己计算类型权重，或者我们自己输入各个类型的权重，比如对于0,1的二元模型，我们可以定义class_weight={0:0.9, 1:0.1}，这样类型0的权重为90%，而类型1的权重为10%。<br/>
如果class_weight选择balanced，那么类库会根据训练样本量来计算权重。某种类型样本量越多，则权重越低，样本量越少，则权重越高。当class_weight为balanced时，类权重计算方法如下：n_samples / (n_classes * np.bincount(y))<br/>
**那么class_weight有什么作用呢?**
在分类模型中，我们经常会遇到两类问题：<br/>
第一种是误分类的代价很高。比如对合法用户和非法用户进行分类，将非法用户分类为合法用户的代价很高，我们宁愿将合法用户分类为非法用户，这时可以人工再甄别，但是却不愿将非法用户分类为合法用户。这时，我们可以适当提高非法用户的权重。<br/>
第二种是样本是高度失衡的，比如我们有合法用户和非法用户的二元样本数据10000条，里面合法用户有9995条，非法用户只有5条，如果我们不考虑权重，则我们可以将所有的测试集都预测为合法用户，这样预测准确率理论上有99.95%，但是却没有任何意义。这时，我们可以选择balanced，让类库自动提高非法用户样本的权重。<br/>
提高了某种分类的权重，相比不考虑权重，会有更多的样本分类划分到高权重的类别，从而可以解决上面两类问题。<br/>
* sample_weight（fit函数参数）.样本权重数<br/>
当样本是高度失衡的，导致样本不是总体样本的无偏估计，从而可能导致我们的模型预测能力下降。遇到这种情况，我们可以通过调节样本权重来尝试解决这个问题。调节样本权重的方法有两种，第一种是在class_weight使用balanced。第二种是在调用fit函数时，通过sample_weight来自己调节每个样本权重。在scikit-learn做逻辑回归时，如果上面两种方法都用到了，那么样本的真正权重是class_weight*sample_weight.<br/>
* max_iter : int, default: 100<br/>
仅在正则化优化算法为newton-cg, sag and lbfgs 才有用，算法收敛的最大迭代次数<br/>
* random_state : int seed, RandomState instance, default: None<br/>
随机数种子，默认为无，仅在正则化优化算法为sag,liblinear时有用
* tol : float, default: 1e-4<br/>
迭代终止判据的误差范围<br/>
* verbose : int, default: 0<br/>
日志冗长度int：冗长度；0：不输出训练过程；1：偶尔输出； >1：对每个子模型都输出<br/>
* warm_start : bool, default: False<br/>
是否热启动，如果是，则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化），bool：热启动，False：默认值<br/>
* n_jobs : int, default: 1<br/>
并行数，int：个数；-1：跟CPU核数一致；1:默认值<br/>

### 常用方法和属性
* decision_function(X) ：Predict confidence scores for samples.
* densify()	： Convert coefficient matrix to dense array format.
* fit(X, y[, sample_weight]) ：训练LR分类器，X是训练样本，y是对应的标记向量
* get_params([deep]) ： Get parameters for this estimator.
* predict(X) ： 预测测试数据，也就是分类。X是测试样本集
* predict_log_proba(X) ： Log of probability estimates.
* predict_proba(X) ： Probability estimates.
* score(X, y[, sample_weight]) ：Returns the mean accuracy on the given test data and labels.
*　set_params(params)　：Set the parameters of this estimator.
* sparsify() : Convert coefficient matrix to sparse format.

## 4、CrossValidation
