# 2023C4_BDC_1

##  环境配置
推荐python版本为3.8.17
安装requirements.txt里面的包即可。

##  数据
使用了官方提供的数据。

##  预训练模型
未使用预训练模型

##  算法
使用Lightgbm模型，采取OVR多分类训练策略进行5折交叉训练。

##  整体思路介绍
由于log和trace中数据的缺失率较高，加之其对线下训练得分的提升帮助不大，我们的方案最终只是用到了metric中的特征。
针对metric数据，将其tags列的数据转化为包含有34个键值对的字典，相当于把这一列数据扩展为了34列。将不同的metric指标按照其tags中含有的不同的键值对进行分类，大致分为三类，分别是4个包含有service_name的指标，15个以container字符开头的指标，94个以node开头的指标，对其中15个以container字符开头的指标和94个以node开头的指标又可以再做进一步的精细化划分，详细划分在代码文件中可以看到。对划分后的不同种类的metric指标联系其特有的键值对生成交叉特征，对不同种类的metric指标进行交叉的tags数量和内容不同。
在复赛过程中，通过输出分类器对每一类故障进行分类时的不同特征的重要性，可以重点关注重要性较高的特征，根据其生成方式生成更多与之类似的交叉特征，也就是让分类器在执行分类的过程中同时执行特征工程，以此达到循环迭代优化特征质量的效果。
最终经过特征筛选后对每个id提取出13259个特征。


##  训练流程
在运行训练脚本train.sh时，
./code/generate_label.py文件会融合各个训练集文件夹下的标签数据，在model/tmp_data/下生成all_train_label_name.csv作为后续训练会用到的标签文件；
./code/feature_extract.py文件会进行特征提取，并在model/tmp_data/下分别生成已经经过筛选的的训练集特征文件ex_f_train_feature.csv与测试集特征文件ex_f_test_feature.csv。
./code/train.py文件会用提取出来的ex_f_train_feature.csv特征拟合分类器，训练得到的分类器会保存在model/model_data文件夹下，五折训练会得到五个LGBM分类器，分别用train_model_kFold1.m,train_model_kFold2.m,train_model_kFold3.m,train_model_kFold4.m,train_model_kFold5.m命名。


##  测试流程
运行推理脚本test.sh，
.test_with_bestmodel.py文件会加载位于model/best_model/文件夹下的产生排行榜结果的的五个模型与已经事先放入test_feature_best/文件夹下的测试集特征文件ex_test_13259feature.csv进行预测,得到产生排行榜结果的预测文件result.csv,位于result/文件夹下。
若是要用训练出来的模型进行推理，则在test.sh中注释掉.test_with_bestmodel.py，同时取消.test_with_newmodel.py的注释，该文件会加载位于model/model_data/文件夹下的五折训练得到的五个模型与已经提取的model/tmp_data文件夹下的测试集的特征文件ex_f_test_feature.csv进行预测（前提是已经完成了train.sh中的特征提取与模型训练）。
