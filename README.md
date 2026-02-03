# Record-Of-My-Postgraduate-Study
这是我的一些踩坑心得

通常我会利用线上的云服务器来复现一些项目，例如Autodl，在这里我会将一些常用的命令总结在这，以及一些常见问题，另外也包括一些比较特别的项目的坑我也会总结在这，欢迎朋友来看我的“记错本”，如有建议，期望一起交流学习~

## 云服务器Autodl踩坑记录

### 1.网络速率问题

网络速率问题，利用autodl自带的学术资源加速指令（解决git不下来的问题）：

可以开学术加速功能，文档里有，命令

```bash
source /etc/network_turbo
```

### 2.完全解决autodl的磁盘管理的问题

必须要弄清楚autodl的磁盘管理的一些基本方法，要不然以后必然寸步难行！

在 AutoDL 上将模型的默认下载路径更改到数据盘，是管理存储空间和避免系统盘爆满的关键操作。下面为清晰、可操作的完整指南。

#### 核心方法：修改环境变量 🛠️

最直接有效的方法是通过设置环境变量，将不同模型库的缓存路径指向数据盘（通常是 `/root/autodl-tmp`）。下表汇总了常用工具的环境变量设置

。

| 模型库 / 工具    | 关键环境变量       | 默认缓存路径 (Linux)    | 修改后的目标路径示例           |
| :--------------- | :----------------- | :---------------------- | :----------------------------- |
| **Hugging Face** | `HF_HOME`          | `~/.cache/huggingface/` | `/root/autodl-tmp/huggingface` |
| **PyTorch**      | `TORCH_HOME`       | `~/.cache/torch/`       | `/root/autodl-tmp/torch`       |
| **ModelScope**   | `MODELSCOPE_CACHE` | 未明确指定              | `/root/autodl-tmp/modelscope`  |

**操作步骤（以 Hugging Face 为例）**

1. **打开配置文件**：通过终端（JupyterLab 或 SSH）登录你的 AutoDL 实例，使用以下命令编辑用户配置文件：

   ```
   vim ~/.bashrc
   ```

2. **添加环境变量**：在文件末尾添加如下行（你可以将 `/root/autodl-tmp/huggingface`替换为你喜欢的任何位于数据盘上的路径）：

   ```
   # 设置 Hugging Face 缓存路径
   export HF_HOME="/root/autodl-tmp/huggingface"
   # 设置 PyTorch 模型缓存路径
   export TORCH_HOME="/root/autodl-tmp/torch"
   # 设置 ModelScope 模型缓存路径
   export MODELSCOPE_CACHE="/root/autodl-tmp/modelscope"
   ```

   同时，强烈建议设置 Hugging Face 镜像以加速下载（本项目不推荐设置这个，会遇到新问题）：

   ```
   export HF_ENDPOINT="https://hf-mirror.com"
   ```

3. **保存并生效**：

   - 在 vim 编辑器中，按 `Esc`键后，输入 `:wq`并按回车来保存并退出。

   - 让配置立即生效：

     ```
     source ~/.bashrc
     ```

   - 你可以通过以下命令检查环境变量是否设置成功：

     ```
     echo $HF_HOME
     ```

     如果终端显示出你设置的路径，说明配置成功。





## pytorch相关问题

Anaconda创建虚拟环境并使用Jupyter-notebook打开虚拟环境

```
https://www.cnblogs.com/xxmmqg/p/12766319.html
```

把环境写进notebook

```bash
python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)"
```



1. 当在用GPU跑一个pytorch的项目时，检查发现：

```text
#cuda是否可用；
torch.cuda.is_available()
```

上述代码检查cuda是否可用，运行结果如下：

```text
out：False
```

2. 然后我们打印一下当前版本的torch，到底下载的是CPU版本的还是GPU版本的

```text
import torch
print(torch.__version__)
```



`python3.8版本以上的要使用conda install nb_conda_kernels命令，否则报错`

这里没有显示[1],而是显示[*]的，可以试试1、pip install ipywidgets 2、jupyter nbextension enable --py widgetsnbextension



打开事件文件：“--port" 指定端口打开避免和服务器里其他人冲突

```bash
tensorboard --logdir=logs --port=6007
```





















## 线下服务器使用技巧

查看并显示时间戳（需先设置HISTTIMEFORMAT）

```bash
export HISTTIMEFORMAT="%F %T "  # 先设置时间格式
history
```

根据您提供的图片信息，**无法直接判断该 `tmp`文件夹是在1月16日创建的**。

图片左侧的资源管理器（文件列表）仅展示了文件和文件夹的名称，**没有显示它们的创建或修改日期**。因此，仅从这张静态截图中，无法获得关于 `tmp`文件夹时间属性的任何信息。

### 🔍 如何准确查看创建时间？

要获取 `tmp`文件夹的精确时间信息，您需要在终端中使用命令。根据图片，这个 `tmp`文件夹位于您的用户主目录中（`~`或 `/home/litong`）。

请在终端中执行以下命令：

1. **查看详细信息（包括最后修改时间）**

   ```
   ls -ld ~/tmp
   ```

   **输出示例**：

   ```
   drwxrwxr-x 2 litong litong 4096 Jan 16 16:47 /home/litong/tmp
   ```

   `Jan 16 16:47`表示该目录内容**最后被修改的时间**。对于新创建的空目录，这个时间通常就是创建时间。

2. **查看更精确的创建时间（如果文件系统支持）**

   ```
   stat ~/tmp
   ```

   这个命令会输出包括**访问时间、修改时间、状态改变时间**在内的详细信息。在某些文件系统（如ext4）上，输出的 `Birth`字段即为创建时间。

### 💡 重要提示：区分两个 “tmp”

结合您之前的历史对话，这一点至关重要：

1. 

   **系统临时目录**：`/tmp`

   这是Linux系统根目录下的标准临时文件夹，由系统在安装时创建，所有用户都可使用。

   您之前遇到的 `pycharm_project_116`很可能就在这个目录下。

2. 

   **用户主目录下的 `tmp`**：`/home/litong/tmp`（即图片中显示的）

   这是您在自己家目录下**手动创建**的普通文件夹，与系统 `/tmp`无关。

   它的创建时间就是您手动创建它的时间。

**总结**：图片中的 `~/tmp`只是一个普通的用户目录，并非系统级 `/tmp`。要确认它的创建时间，必须使用 `ls -ld`或 `stat`命令在终端中查看。















