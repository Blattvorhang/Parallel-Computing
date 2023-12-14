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
由调用可执行文件时的命令行参数给出。

目前能想到的优化方案：
1. 多线程：对数组采用二分，每二分一次产生两个线程，由于`MAX_THREADS`为64，递归深度为5（1+2+4+8+16+32=63）
2. SSE（单指令多数据）：可能要提前进行代码框架设计
3. CUDA：和SSE类似，要提前进行代码框架设计