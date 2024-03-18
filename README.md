## Issues

### 1 DataLoader 的多进程 pickle

<img src="https://raw.githubusercontent.com/PLUS-WAVE/blog-image/master/img/blog/2024-03-17/image-20240317200544474.png" alt="image-20240317200544474" style="zoom: 50%;" />

在刚开始执行时就遇到了这个错误，开始以为是环境问题，结果鼓捣半天没用，仔细分析了一下错误原因，发现主要是这句话的原因

```python
for iteration, batch in enumerate(data_loader):
```

也就是序列化的问题，后面还给出了多进程的保持，就应该是**多进程的pickle**问题，网上一搜，还真是，可能是Windows系统的原因导致的，在 `configs` 中的配置文件中修改 `num_workers = 0`，不使用多进程，就解决了报错



### 2 imageio 输出图片

在执行到 `evaluate` 时，输出图片出了下面的错误：

> `envs\NeRFlearning\Lib\site-packages\PIL\Image.py", line 3102, in fromarray raise TypeError(msg) from e TypeError: Cannot handle this data type: (1, 1, 3), <f4`

这是因为`imageio.imwrite`函数需要接收的图像数据类型为`uint8`，而原始的 `pred_rgb` 和 `gt_rgb` 可能是浮点数类型的数据。因此，我们需要将它们乘以255（将范围从0-1转换为0-255），然后使用 `astype` 函数将它们转换为 `uint8` 类型

```python
# 将数据类型转换为 uint8
pred_rgb = (pred_rgb * 255).astype(np.uint8)
gt_rgb = (gt_rgb * 255).astype(np.uint8)
# 需要添加以上两行，否则报错
imageio.imwrite(save_path, img_utils.horizon_concate(gt_rgb, pred_rgb))
```

这里还有一个问题，我发现在已经生成过一张图片后，再次执行到这里，输出的新图片无法覆盖之前的旧图片，所以我加上了时间和当前周期作为扩展名

```python
now = datetime.datetime.now()
now_str = now.strftime('%Y-%m-%d_%Hh%Mm%Ss')
base_name, ext = os.path.splitext(save_path)
save_path = f"{base_name}_{now_str}_epoch_{cfg.train.epoch}{ext}"
```

这样就会以这样的 `res_2024-03-18_09h09m40s_epoch_10.jpg` 格式正确输出每次的图像了



### 3 I/O

此项目是由 Linux 开发的，在Windows系统上，免不了出现各种麻烦。特别是，该项目的所有 I/O 都是 Linux 的 I/O 格式，所以要进行全面修改：

- 使用 Python 的 os 模块中的 `makedirs` 函数来替换 

  ```python
  os.system('mkdir -p ' + model_dir)
  # |
  # V
  os.makedirs(model_dir, exist_ok=True)
  ```

- `shutil` 模块中的 `rmtree` 函数来替换

  ```python
  os.system('rm -rf {}'.format(model_dir))
  # |
  # V
  import shutil
  if os.path.exists(model_dir):
  	shutil.rmtree(model_dir)
  ```

- Python 的 os 模块中的`remove`函数来替换

  ```python
  os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))
  # |
  # V
  os.remove(os.path.join(model_dir, '{}.pth'.format(min(pths))))
  ```

- 使用 `exit(0)` 来结束当前进程

  ```python
  os.system('kill -9 {}'.format(os.getpid()))
  # |
  # V
  exit(0)
  ```

  

## Tutorial

### Data preparation

Download NeRF synthetic dataset and add a link to the data directory. After preparation, you should have the following directory structure: 
```
data/nerf_synthetic
|-- chair
|   |-- test
|   |-- train
|   |-- val
|-- drums
|   |-- test
......
```


### 从Image fitting demo来学习这个框架


#### 任务定义

训练一个MLP，将某一张图像的像素坐标作为输入, 输出这一张图像在该像素坐标的RGB value。

#### Training

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```

#### Evaluation

```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```

#### 查看loss曲线

```
tensorboard --logdir=data/record --bind_all
```


### 开始复现NeRF

#### 配置文件

我们已经在configs/nerf/ 创建好了一个配置文件，nerf.yaml。其中包含了复现NeRF必要的参数。
你可以根据自己的喜好调整对应的参数的名称和风格。


#### 创建dataset： lib.datasets.nerf.synthetic.py

核心函数包括：init, getitem, len.

init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式。

getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。
例如对NeRF，分别是1024条rays以及1024个RGB值。

len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]。


##### debug：

```
python run.py --type dataset --cfg_file configs/nerf/nerf.yaml
```

#### 创建network:

核心函数包括：init, forward.

init函数负责定义网络所必需的模块，forward函数负责接收dataset的输出，利用定义好的模块，计算输出。例如，对于NeRF来说，我们需要在init中定义两个mlp以及encoding方式，在forward函数中，使用rays完成计算。


##### debug：

```
python run.py --type network --cfg_file configs/nerf/nerf.yaml
```

#### loss模块和evaluator模块

这两个模块较为简单，不作仔细描述。

debug方式分别为：

```
python train_net.py --cfg_file configs/nerf/nerf.yaml
```

```
python run.py --type evaluate --cfg_file configs/nerf/nerf.yaml
```
