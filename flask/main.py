from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
import json
app = Flask(__name__)
import sys
import os
sys.path.append("..")
from data_message.arxiv_api import arxiv_find_key_paper
from data_message.data_find import data_predict, data_index
import heapq

data_options = {
}
show_options = {
}
monthes=[]


@app.route('/')
def index():
    return render_template('index.html', data_options=data_options.keys() ,show_options=show_options.keys())

@app.route('/get_data', methods=['POST'])
def get_data():
    data = {'data':data_options,
            'months':monthes}
    return jsonify(data)
@app.route('/get_Analysis_text', methods=['POST'])
def get_Analysis_text():
    data = {'data':show_options}
    return jsonify(data)

@app.route('/get_key_text', methods=['POST'])
def get_key_text():
    data = request.get_json()
    key = data.get('key')
    print(key)
    key = key.lower()
    paper_path=f'/mnt/data2/jingxz/system/data_message/store_keys/{key}_papers_counts.xlsx'
    author_path = f'/mnt/data2/jingxz/system/data_message/store_keys/{key}_authors_counts.json'
    if not os.path.exists(paper_path) or not os.path.exists(author_path):
        data_index(key)
    predict_path=f'/mnt/data2/jingxz/system/data_message/predicts/{key}.xlsx'
    if not os.path.exists(f'/mnt/data2/jingxz/system/data_message/predicts/{key}.xlsx'):
        data_predict(key)
    df_row = pd.read_excel(predict_path)
    monthes = df_row['monthes']
    paper = df_row['papers']
    papers = {'months':list(monthes), 'data':list(paper)}
    #所有方向
    cols = list(df_row.columns)
    cols.remove('monthes')
    cols.remove('papers')
    #学科
    subject={}
    for col in cols:
        subject[col]=list(df_row[col])
    subject={'months':list(monthes),
            'data':subject}

    with open(author_path, "r", encoding="utf-8") as f:
        author_data = json.load(f)  # 返回字典或列表
    # print(author_data.keys())
    author_dict = author_data['author_dict']
    author_link = author_data['author_link_data']
    category = author_data['category']

    top_10 = heapq.nlargest(50, author_dict.items(), key=lambda x: x[1])
    print(jsonify({'papers':papers,
            'author':top_10,
            'subject':subject,
            'author_link':author_link,
            'category':category}))
    return jsonify({'papers':papers,
            'author':top_10,
            'subject':subject,
            'author_link':author_link,
            'category':category})

@app.route('/arxiv_api_papers', methods=['POST'])
def arxiv_api_papers():
    data = request.get_json()
    key = data.get('key')
    paper_text=arxiv_find_key_paper(key)
    return jsonify({'papers': paper_text})



def get_month():
    return jsonify(monthes)

def get_predcit():
    #数据信息
    data_path = '../data_message/predicts.xlsx'
    df_row = pd.read_excel(data_path)
    cols = list(df_row.columns)
    cols.remove('monthes')
    global monthes
    global data_options
    monthes=list(df_row['monthes'])
    for name in cols:
        # data_options[name]=df_row[name]
        if 'paper' in name:
            t_name = name.removesuffix('_papers')
            if t_name in data_options:
                data_options[t_name]['paper'] = list(df_row[name])
            else:
                data_options.update({t_name:{'paper':list(df_row[name])}})
        elif 'authors' in name:
            t_name = name.removesuffix('_authors')
            if t_name in data_options:
                data_options[t_name]['author'] = list(df_row[name])
            else:
                data_options.update({t_name:{'author':list(df_row[name])}})
    #gpt得到的文本信息
    text_path = '../data_message/analysis_data.jsonl'
    global show_options
    with open(text_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            show_options.update({item.get('name'):item.get('text')})
    



if __name__ == '__main__':
    get_predcit()
    app.run(debug=True,port=3636)
    



