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
