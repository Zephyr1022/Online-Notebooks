### [23 种 Pandas 核心操作，你需要过一遍吗？](https://www.jiqizhixin.com/articles/082602)

在本文中，基本数据集操作主要介绍了 CSV 与 Excel 的读写方法，基本数据处理主要介绍了缺失值及特征抽取，最后的 **DataFrame** 操作则主要介绍了函数和排序等方法。

https://towardsdatascience.com/23-great-pandas-codes-for-data-scientists-cca5ed9d8a38

[More](https://towardsdatascience.com/23-great-pandas-codes-for-data-scientists-cca5ed9d8a38)

### 1, Data Check and reader

```python
# 读取 CSV 格式的数据集
# Read in a CSV dataset
pd.DataFrame.from_csv(“csv_file”)
pd.read_csv(“csv_file”)

# 读取 Excel 数据集
pd.read_excel("excel_file")

# 将 DataFrame 直接写入 CSV 文件, 如下采用逗号作为分隔符，且不带索引：
# Write your data frame directly to csv, Write your data frame directly to csv
df.to_csv("data.csv", sep=",", index=False)

# 基本的数据集特征信息,
# Basic dataset feature info
df.info() 

#基本的数据集统计信息
# Basic dataset statistics
print(df.describe())

# Print data frame in a table, 将 DataFrame 输出到一张表：
# 当「print_table」是一个列表，其中列表元素还是新的列表，「headers」为表头字符串组成的列表。
# where “print_table” is a list of lists and “headers” is a list of the string headers
print(tabulate(print_table, headers=headers))

# 列出所有列的名字
#  List the column names
df.columns
```



### 2, Data Cleaning & Handling

```python
# 基本数据处理 #
# 删除缺失数据
# 返回一个DataFrame,其中删除了包含任何 NaN 值的给定轴，选择 how=「all」会删除所有元素都是 NaN 的给定轴。
# Drop missing data
# Returns object with labels on given axis omitted where alternately any or all of the data are missing
df.dropna(axis=0, how='any')

# 替换缺失数据
# Replace missing data
# 使用 value 值代替 DataFrame 中的 to_replace 值，其中 value 和 to_replace 都需要我们赋予不同的值。
# replaces values given in “to_replace” with “value”.
df.replace(to_replace=None, value=None)

# 检查空值 NaN
# Check for NANs
# 检查缺失值，即数值数组中的 NaN 和目标数组中的 None/NaN。
# Detect missing values (NaN in numeric arrays, None/NaN in object arrays)
pd.isnull(object)

# 删除特征
# Drop a feature
# axis is either 0 for rows, 1 for columns
# axis 选择 0 表示行，选择表示列。
df.drop('feature_variable_name', axis=1)

# 将目标类型转换为浮点型
# Convert object type to float
# Convert object types to numeric to be able to perform computations (in case they are string)
# 将目标类型转化为数值从而进一步执行计算，在这个案例中为字符串。
pd.to_numeric(df["feature_name"], errors='coerce')

# 将 DataFrame 转换为 NumPy 数组
# Convert data frame to numpy array
df.as_matrix()

# 取 DataFrame 的前面「n」行
# Get first “n” rows of a data frame
df.head(n)

# 通过特征名取数据
# Get data by feature name
df.loc[feature_name]
```



### 3, Operating on data frames

```python
# DataFrame 操作 #
# 该函数将令 DataFrame 中「height」列的所有值乘上 2：
# Apply a function to a data frame
# This one will multiple all values in the “height” column of the data frame by 2
df["height"].apply(*lambda* height: 2 * height)
# 或：
def multiply(x):
	return x * 2
df["height"].apply(multiply)

# 重命名列
# Renaming a column
# Here we will rename the 3rd column of the data frame to be called “size”
# 下面代码会重命名 DataFrame 的第三列为「size」：
df.rename(columns = {df.columns[2]:'size'}, inplace=True)

#取某一列的唯一实体
#下面代码将取「name」列的唯一实体：
# Get the unique entries of a column
# Here we will get the unique entries of the column “name”
df["name"].unique()

# 访问子 DataFrame
# Accessing sub-data frames
# Here we’ll grab a selection of the columns, “name” and “size” from the data frame
# 以下代码将从 DataFrame 中抽取选定了的列「name」和「size」：
new_df = df[["name", "size"]]

x_train = pd.read_csv(‘data_input.csv’,header=None) #数据类型为dataframe
x_train.values #直接转成矩阵
```



### 4, Summary information about your data

```PYTHON
# 总结数据信息
# Sum of values in a data frame
df.sum()
# Lowest value of a data frame
df.min()
# Highest value
df.max()
# Index of the lowest value
df.idxmin()
# Index of the highest value
df.idxmax()
# Statistical summary of the data frame, with quartiles, median, etc.
df.describe()
# Average values
df.mean()
# Median values
df.median()
# Correlation between columns
df.corr()
# To get these values for only one column, just select it like this#
df["size"].median()
```



```PYTHON
# Sorting your data
df.sort_values(ascending = False)

#布尔型索引
#以下代码将过滤名为「size」的列，并仅显示值等于 5 的列：
# Boolean indexing
# Here we’ll filter our data column named “size” to show only values equal to 5
df[df["size"] == 5]

# 选定特定的值
# 以下代码将选定「size」列、第一行的值：
# Selecting values
# Let’s select the first row of the “size” column
df.loc([0], ['size'])
```

