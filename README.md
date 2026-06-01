# system README

本项目是一个基于 arXiv 数据的学科趋势分析系统，主要包含三个部分：

- `data_message/`：数据清洗、关键词检索、学科分析
- `MixF/`：时序预测模型与预测脚本
- `flask/`：Flask 后端接口
- `flask/my-app/`：Next.js 前端页面

## 目录说明

### 1. data_message

主要功能：

- 将 arXiv 原始数据转成 `jsonl`
- 根据关键词统计论文、作者和学科方向信息
- 调用大模型生成学科分析文本

关键文件：

- `data_message/json2jsonl.py`
- `data_message/data_find.py`
- `data_message/arxiv_api.py`

### 2. MixF

主要功能：

- 训练 / 加载 MixF 模型
- 对学科趋势数据进行预测
- 生成预测结果文件

关键文件：

- `MixF/run.py`
- `MixF/scripts/arxiv_predict.sh`
- `MixF/scripts/key_predict.sh`
- `MixF/dataset/json2excel.py`

### 3. flask

主要功能：

- 提供数据接口
- 为前端返回学科趋势、关键词分析和论文信息

关键文件：

- `flask/main.py`

默认后端地址：

- `http://127.0.0.1:3636`

### 4. flask/my-app

主要功能：

- 提供 Next.js 前端页面
- 展示学科趋势、关键词分析和论文信息

关键文件：

- `flask/my-app/src/app/(subject)/page.tsx`
- `flask/my-app/src/app/key-word/page.tsx`
- `flask/my-app/package.json`

默认前端地址：

- `http://127.0.0.1:3000`

## 运行流程

建议在项目根目录执行：

```bash
cd /mnt/data2/jingxz/system
```

### 1. 准备原始数据

原始 arXiv 数据需要放在：

- `MixF/dataset/arxiv-metadata-oai-snapshot.json`

### 2. 生成结构化数据

```bash
python data_message/json2jsonl.py
```

### 3. 生成 MixF 输入数据

```bash
cd MixF/dataset
python json2excel.py
cd /mnt/data2/jingxz/system
```

生成后会得到：

- `MixF/dataset/categories.xlsx`

### 4. 运行预测模型

```bash
cd MixF
bash scripts/arxiv_predict.sh
cd /mnt/data2/jingxz/system
```

### 5. 生成大模型分析结果

运行前需要先设置环境变量：

```bash
export GPTGOD_API_KEY=your_api_key
```

然后执行：

```bash
python data_analysis.py
```

### 6. 启动 Web 服务

项目需要同时启动两个程序：

- Flask 后端：`flask/main.py`
- Next.js 前端：`flask/my-app` 的 `npm run dev`

推荐在项目根目录执行：

```bash
bash start.sh
```

启动后访问：

- 前端页面：`http://127.0.0.1:3000`
- 后端接口：`http://127.0.0.1:3636`

`start.sh` 会在后台启动 Flask 后端，然后在前台启动 Next.js 前端。停止脚本时会自动关闭本次启动的 Flask 后端进程。

如果需要手动启动，可以分别打开两个终端执行：

终端 1：

```bash
cd flask
python main.py
```

终端 2：

```bash
cd flask/my-app
npm run dev
```

## 一键流程

项目中提供了 Web 服务启动脚本：

- `start.sh`

执行方式：

```bash
bash start.sh
```

如果需要重新生成数据、运行 MixF 预测或生成大模型分析结果，请按上面的“运行流程”第 1-5 步执行。`start.sh` 只负责启动后端和前端服务，不会下载数据、训练模型或调用大模型 API。

## 依赖说明

至少需要这些 Python 依赖：

- `flask`
- `pandas`
- `numpy`
- `requests`
- `aiohttp`
- `aiofiles`
- `tqdm`
- `openpyxl`
- `arxiv`
- `torch`

如果需要运行 `MixF`，还需要准备对应的深度学习环境。

前端至少需要：

- Node.js
- npm

首次启动时，如果 `flask/my-app/node_modules` 不存在，`start.sh` 会自动执行 `npm install`。

## 说明

- 本仓库默认不上传大体积原始数据、训练日志、checkpoint 和缓存文件。
- `flask/my-app/` 是当前主线前端入口，需要和 Flask 后端一起启动。
