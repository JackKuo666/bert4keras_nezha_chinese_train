# 使用bert4keras 对nezha 进行训练


## 依赖
```py
tensorflow-gpu            2.4.0 
tensorboard               2.9.0
bert4keras                0.11.3 
```

## 原理

1.数据可以使用`lazy`加载，不过得自己计算好总的步数，设置`step_per_epoch`

2.nezha使用`jieba`加载自定义词典训练

3.模型的加载与保存、checkpoint的设置、earlystop 的设置