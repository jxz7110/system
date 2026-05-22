conda activate torch
#下载数据
cd /mnt/data2/jingxz/system/MixF/dataset
bash ./arxiv_download.sh
# #将数据量化转化为excel文件保存
python ./json2excel.py
rm -f /mnt/data2/jingxz/system/data_message/data.jsonl
python /mnt/data2/jingxz/system/data_message/json2jsonl.py
# 训练模型
cd /mnt/data2/jingxz/system/MixF
# #将输入带入模型MixF中进行训练并保存checkpoint,同时将预测数据保存在predicts.xlsx文件中
bash ./scripts/arxiv_predict.sh

#重新生成大模型api分析结果
rm -f /mnt/data2/jingxz/system/data_message/analysis_data.jsonl
cd /mnt/data2/jingxz/system/
python /mnt/data2/jingxz/system/data_analysis.py
# 通过输入数据生成分析结果
# 删除以往缓存的数据
rm -f /mnt/data2/jingxz/system/data_message/store_keys/*
rm -f /mnt/data2/jingxz/system/data_message/predicts/*

