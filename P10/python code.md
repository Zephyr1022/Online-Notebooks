# Python中字典创建的几种方法及适用场景



https://blog.csdn.net/Jerry_1126/article/details/78239530

**方式一: 直接创建**

```python
D1 = {'name': 'Tom', 'age': 40}           # 直接创建
D1                                        # 适用场景: 事先已经拼接出整个字典
{'age': 40, 'name': 'Tom'}
```

***\*方式二: 动态创建\**** 

```
D2 = {}                                   # 动态创建
D2['name'] = 'Tom'                        # 动态赋值
D2['age'] = 40		              # 动态赋值
D2                                        # 适用场景: 适用于动态创建字典的一个字段
{'age': 40, 'name': 'Tom'}
```

***\*方式三: 关键字创建\**** 

```
D4 = dict((['name', 'Tom'], ['age', 40])) # 键值对创建
D4                                        # 适用场景: 需要把键值逐步建成序列，此形式比较适用
{'age': 40, 'name': 'Tom'}
```

#### Python 字典(Dictionary) fromkeys()方法

https://www.runoob.com/python/att-dictionary-fromkeys.html





#### Python小白需要知道的 20 个骚操作！	

https://zhuanlan.zhihu.com/p/87787535

