# 12.13 (Blattvorhang)
基本思路：
注意max, sum, sort三个操作都可以进行拆分，由于不涉及修改操作，直接使用二分法来分配算力。

关键点：  
1. 拆分后的合并
   1. max和sum的合并都很直观，sort采用mergeSort的merge操作来实现。
2. 该拆多少，每部分要拆到哪
3. 是否选择GPU（后续考虑）

## CMake的构建
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## 文件执行
```bash
./SpeedUp_Computing
```

## 修改内容
建立项目基本文件架构，CMakeLists.txt设置完成，建立两个bash文件，其中`build.sh`用于编译构建可执行文件，`run.bash`用于执行生成的文件。

文件说明：
```shell
.
├── CMakeLists.txt      # 项目构建文件
├── LICENSE             # MIT License
├── README.md
├── build               # 编译生成文件夹，.gitignore已忽略
├── build.bash          # 编译脚本
├── docs
│   └── CHANGELOG.md    # 开发日志
├── include             # 头文件
│   └── calculation.h
├── run.bash            # 运行脚本
└── src                 # CPP源文件
    ├── common.cpp      # 公用函数
    ├── main.cpp        # 主函数
    ├── original.cpp    # 加速前
    └── speedup.cpp     # 加速后
```

添加了宏函数，表示对数据的访问：
```cpp
#define ACCESS(data) log(sqrt(data))
```

# 12.14 (Blattvorhang)
完成了`original.cpp`，区分`client`与`server`，`client`发出信号（暂定为数组起始下标）给`server`，`server`计算完成后把结果发回`client`。
三种运行模式：
1. LOCAL
2. CLIENT
3. SERVER

模式选择由调用可执行文件时的命令行参数给出。

目前能想到的优化方案：
1. 多线程：对数组采用二分，每二分一次产生两个线程，由于`MAX_THREADS`为64，递归深度为5（1+2+4+8+16+32=63）
2. SSE（单指令多数据）：可能要提前进行代码框架设计
3. CUDA：和SSE类似，要提前进行代码框架设计


# 12.15 (Blattvorhang)
求和变量`sum_value`如果选用`float`会产生精度误差，串行和并行结果不一致，其中一次结果如下：
```
Running in local mode.

--- Original version ---
Time consumed: 5.62687s
sum: 2.68435e+08
max: 9.33377

--- Speedup version ---
Time consumed: 0.753798s
sum: 1.12936e+09
max: 9.33377

Speedup ratio: 7.46469
```
由于log(sqrt(data))的输入和返回参数都是`double`类型，主要耗时在这两个函数而非加法上，故加法采用`double`并不会产生较大的时间损失。
`sum_value`改用`double`类型后，结果如下：
```
Running in local mode.

--- Original version ---
Time consumed: 5.56075s
sum: 1.13072e+09
max: 9.33377

--- Speedup version ---
Time consumed: 0.704421s
sum: 1.13072e+09
max: 9.33377

Speedup ratio: 7.89407
```
可以发现用时差别不大（好像还变快了，怀疑是前面`float`转`double`花了额外的时间），而加速前后的结果不再有差别。

# 12.16 (Blattvorhang)
并行化归并排序思路：
递归地把数组拆成`MAX_THREADS`份进行归并排序，随后两两合并，排序和合并均并行执行。考虑到合并过程也占用线程，应该拆成`MAX_THREADS / 2`份。
实测CPU跑满，结果如下：
```
--- Original version ---
sum time consumed: 1.3555s
max time consumed: 1.49854s
sort time consumed: 108.965s
Time consumed: 111.82s
sum: 1.13072e+09
max: 9.33377
Result is sorted.

--- Speedup version ---
sum time consumed: 0.295863s
max time consumed: 0.314429s
sort time consumed: 23.6107s
Time consumed: 24.2212s
sum: 1.13072e+09
max: 9.33377
Result is sorted.

Speedup ratio: 4.61661
```
加速比并未达到理想中的7左右，怀疑是归并排序复制数据部分占用了太多时间，可以考虑更换排序算法。

于是我测试性地把自己写的归并排序改成了`std::sort`，分成64个部分并行地排序，最后合并起来，结果如下：
```
--- Original version ---
sum time consumed: 1.2942s
max time consumed: 1.44973s
sort time consumed: 103.683s
Time consumed: 106.427s
sum: 1.13072e+09
max: 9.33377
Result is sorted.

--- Speedup version ---
sum time consumed: 0.227853s
max time consumed: 0.282442s
sort time consumed: 12.6314s
Time consumed: 13.1418s
sum: 1.13072e+09
max: 9.33377
Result is sorted.

Speedup ratio: 8.0983
```
所以确实可以考虑在排序上面下功夫。

# 12.17 (Blattvorhang)
对`merge`过程做了优化。由于是并行地对数组进行排序，最后还要有一个`merge`过程把子数组合并为一整个有序的数组，这个过程涉及大量数据的复制，之前是用一个`temp`数组来存，先合并到`temp`中，再写回`arr`。举例如下：
```
把一个数组A拆分为八个部分：
A1 A2 A3 A4 A5 A6 A7 A8
分别对每个部分排好序后，合并到数组B中：
B12 B34 B56 B78
接下来把B复制到A中：
A12 A34 A56 A78
随后再把A继续合并到B：
B1234 B5678
接下来把B复制到A中：
A1234 A5678
把A合并到B：
B12345678
把B复制到A中：
A12345678
```
上面的例子充分说明了时间的浪费，故现在考虑两个数组`arr1`和`arr2`，交错地来作为存结果的数组，即：
```
把一个数组A拆分为八个部分：
A1 A2 A3 A4 A5 A6 A7 A8
分别对每个部分排好序后，合并到数组B中：
B12 B34 B56 B78
接下来把B合并到A中：
A1234 A5678
把A合并到B中：
B12345678
最后一步把B复制回A即可：
A12345678
```
可以发现，先前需要6步的合并过程，现在优化到只需要4步了，对于`MAX_THREAD = 64`的情况，带来的优化只会更加明显。但是需要注意什么情况下需要把B复制回A，什么情况不需要，这需要对层数`level`的奇偶性进行判断。

优化后的数据如下：
```
--- Original version ---
Sum time consumed: 1.86367
Max time consumed: 2.23014
Sort time consumed: 153.145
Total time consumed: 157.239

sum: 1.13072e+09
max: 9.33377
Result is sorted.

--- Speedup version ---
Sum time consumed: 0.283588
Max time consumed: 0.310064
Sort time consumed: 25.9752
Total time consumed: 26.5689

sum: 1.13072e+09
max: 9.33377
Result is sorted.

Sum speedup ratio: 6.57173
Max speedup ratio: 7.19251
Sort speedup ratio: 5.89581
Total speedup ratio: 5.91816
```

# 12.20 (Blattvorhang)
开启两个终端，一个以client运行，一个以server运行，用回环地址`127.0.0.1:8080`连接成功。