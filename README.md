# system README

本项目是一个基于 arXiv 数据的学科趋势分析系统，主要包含三个部分：

- `data_message/`：数据清洗、关键词检索、学科分析
- `MixF/`：时序预测模型与预测脚本
- `flask/`：后端接口与可视化页面

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

- 提供网页接口
- 展示学科趋势、关键词分析和论文信息

关键文件：

- `flask/main.py`
- `flask/templates/index.html`

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

### 6. 启动 Flask 页面

```bash
cd flask
python main.py
```

默认访问地址：

- `http://127.0.0.1:3636`

## 一键流程

项目中提供了一个简单流程脚本：

- `start.sh`

执行方式：

```bash
bash start.sh
```

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

## 说明

- 本仓库默认不上传大体积原始数据、训练日志、checkpoint 和缓存文件。
- `flask/my-app/` 是本地前端工作目录，不作为当前主线运行入口。
