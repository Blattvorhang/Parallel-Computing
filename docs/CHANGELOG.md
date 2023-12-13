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
./Parallel_Accelerated_Computing
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