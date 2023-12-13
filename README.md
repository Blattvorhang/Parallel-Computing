# Parallel-Accelerated-Computing

# 开发日志
[CHANGELOG.md](./docs/CHANGELOG.md)

# 项目内容
两人一组，利用相关C++需要和加速(sse，多线程)手段，以及通讯技术(1.rpc，命名管道，2.http，socket)等实现函数（浮点数数组求和，求最大值，排序）。所有处理在两台计算机协作执行，尽可能挖掘两个计算机的潜在算力。

## 要求
书写完整报告，给出设计思路和结果。特别备注当我重现你们代码时，需要修改的位置和含义，精确到文件的具体行。

## 提交材料
报告和程序。

## 给分细则
加速比越大分数越高；多人组队的，分数低于同加速比的两人组分数；非windows上实现加5分（操作系统限于ubuntu,android）；

## 备注
报告中列出执行5次任务，并求时间的平均值。需要附上两台单机原始算法执行时间，以及利用双机并行执行同样任务的时间。

## 特别说明
最后加速比以我测试为准。报告中的时间统计必须真实有效，发现舞弊者扣分。如果利用超出我列举的技术和平台，先和我商议。

## 追加
三个功能自己代码实现，不得调用第三方库函数（比如，`std::max`,`std::sort`）,违反者每函数扣10分。多线程，多进程，OPENMP可以使用。

## 数据
自己生成，可参照下述方法。

测试数据量和计算内容为：

```cpp
#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /*这个数值是总数据量*/

//待测试数据定义为：
float rawFloatData[DATANUM];
```

参照下列数据初始化代码：两台计算机可以分开初始化，减少传输代价。
```cpp
for (size_t i = 0; i < DATANUM; i++)//数据初始化
{
	rawFloatData[i] = float(i+1);
}
```

为了模拟任务：每次访问数据时，用`log(sqrt(rawFloatData[i]))`进行访问！就是说比如计算加法，用 `sum+=log(sqrt(rawFloatData[i]))`,而不是`sum+=rawFloatData[i]` !!。这里计算结果和存储精度之间有损失，但你们机器的指令集限制，如果使用SSE中的`double`型的话，单指令只能处理4个`double`，如果是`float`则可以8个。所以用`float`加速比会更大。

## 代码要求
### 提供代码（1/2）  
需要提供的函数：(不加速版本，为同样的数据量在两台计算机上各自运行的时间。算法一致，只是不采用任何加速手段（SSE，多线程或者OPENMP)）

```cpp
float sum(const float data[], const int len); //data是原始数据，len为长度。结果通过函数返回
float max(const float data[], const int len);//data是原始数据，len为长度。结果通过函数返回
float sort(const float data[], const int len, float result[]);//data是原始数据，len为长度。排序结果在result中。
```

### 提供代码（2/2）双机加速版本  
```cpp
float sumSpeedUp(const float data[], const int len); //data是原始数据，len为长度。结果通过函数返回
float maxSpeedUp(const float data[], const int len);//data是原始数据，len为长度。结果通过函数返回
float sortSpeedUp(const float data[], const int len, float result[]);//data是原始数据，len为长度。排序结果在result中。
```
加速中如果使用SSE，特别注意SSE的指令和数据长度有关，单精度后缀ps,双精度后缀pd。

测试速度的代码框架为：
```cpp
QueryPerformanceCounter(&start);//start  
你的函数(...);//包括任务启动，结果回传，收集和综合
QueryPerformanceCounter(&end);//end
std::cout << "Time Consumed:" << (end.QuadPart - start.QuadPart) << endl;
cout<<输出求和结果<<endl;
cout<<输出最大值<<endl;
cout<<输出排序是否正确<<endl;
```

注意，如果单机上那么大数据量无法计算，可以只算一半。修改 `#define SUBDATANUM 2000000` 为 `#define SUBDATANUM 1000000`做单机计算。双机上每个计算机都申请`#define SUBDATANUM 1000000`大的数据，即实现`#define SUBDATANUM 2000000`的运算。