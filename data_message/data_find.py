import numpy as np
import copy
# import pickle as pkl
import json
import os
from collections import defaultdict
import pandas as pd
import subprocess
from tqdm import tqdm
from datetime import datetime,timedelta
import argparse
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy 
import calendar
from collections import defaultdict, OrderedDict


subject_label ={
             'cs':'Computer Science',
             'econ':'Economics',
             'eess':'Electrical Engineering and Systems Science',
             'math':'Mathematics',
             'astro-ph':'Physics', 'cond-mat':'Physics', 'gr-qc':'Physics', 'hep-ex':'Physics', 'hep-lat':'Physics', 'hep-ph':'Physics', 'hep-th':'Physics', 'math-ph':'Physics', 'nlin':'Physics', 'nucl-ex':'Physics', 'nucl-th':'Physics','physics':'Physics', 'quant-ph':'Physics',
             'q-bio':'Quantitative Biology',
             'q-fin':'Quantitative Finance',
             'stat':'Statistics'
}
direction_label={
    'Computer Science':{
        'AI':'Artificial Intelligence',
        'AR':'Hardware Architecture',
        'CC':'Computational Complexity',
        'CE':'Computational Engineering, Finance, and Science',
        'CG':'Computational Geometry',
        'CL':'Computation and Language',
        'CR':'Cryptography and Security',
        'CV':'Computer Vision and Pattern Recognition',
        'CY':'Computers and Society',
        'DB':'Databases',
        'DC':'Distributed, Parallel, and Cluster Computing',
        'DL':'Digital Libraries',
        'DM':'Discrete Mathematics',
        'DS':'Data Structures and Algorithms',
        'ET':'Emerging Technologies',
        'FL':'Formal Languages and Automata Theory',
        'GL':'General Literature',
        'GR':'Graphics',
        'GT':'Computer Science and Game Theory',
        'HC':'Human-Computer Interaction',
        'IR':'Information Retrieval',
        'IT':'Information Theory',
        'LG':'Machine Learning',
        'LO':'Logic in Computer Science',
        'MA':'Multiagent Systems',
        'MM':'Multimedia',
        'MS':'Mathematical Software',
        'NA':'Numerical Analysis',
        'NE':'Neural and Evolutionary Computing',
        'NI':'Networking and Internet Architecture',
        'OH':'Other Computer Science',
        'OS':'Operating Systems',
        'PF':'Performance',
        'PL':'Programming Languages',
        'RO':'Robotics',
        'SC':'Symbolic Computation',
        'SD':'Sound',
        'SE':'Software Engineering',
        'SI':'Social and Information Networks',
        'SY':'Systems and Control'
    },
    'Economics':{
        'EM':'Econometrics',
        'GN':'General Economics',
        'TH':'Theoretical Economics',
    },
    'Electrical Engineering and Systems Science':{
        'AS':'Audio and Speech Processing',
        'IV':'Image and Video Processing',
        'SP':'Signal Processing',
        'SY':'Systems and Control'
    },
    'Mathematics':{
        'AC':'Commutative Algebra',
        'AG':'Algebraic Geometry',
        'AP':'Analysis of PDEs',
        'AT':'Algebraic Topology',
        'CA':'Classical Analysis and ODEs',
        'CO':'Combinatorics',
        'CT':'Category Theory',
        'CV':'Complex Variables',
        'DG':'Differential Geometry',
        'DS':'Dynamical Systems',
        'FA':'Functional Analysis',
        'GM':'General Mathematics',
        'GN':'General Topology',
        'GR':'Group Theory',
        'GT':'Geometric Topology',
        'HO':'History and Overview',
        'IT':'Information Theory',
        'KT':'K-Theory and Homology',
        'LO':'Logic',
        'MG':'Metric Geometry',
        'MP':'Mathematical Physics',
        'NA':'Numerical Analysis',
        'NT':'Number Theory',
        'OA':'Operator Algebras',
        'OC':'Optimization and Control',
        'PR':'Probability',
        'QA':'Quantum Algebra',
        'RA':'Rings and Algebras',
        'RT':'Representation Theory',
        'SG':'Symplectic Geometry',
        'SP':'Spectral Theory',
        'ST':'Statistics Theory',
    },
    'Physics':{
        'astro-ph':'Astrophysics',
        'cond-mat':'Condensed Matter',
        'CO':'Cosmology and Nongalactic Astrophysics',
        'EP':'Earth and Planetary Astrophysics',
        'GA':'Astrophysics of Galaxies',
        'HE':'High Energy Astrophysical Phenomena',
        'IM':'Instrumentation and Methods for Astrophysics',
        'SR':'Solar and Stellar Astrophysics',
        'dis-nn':'Disordered Systems and Neural Networks',
        'mes-hall':'Mesoscale and Nanoscale Physics',
        'mtrl-sci':'Materials Science',
        'other':'Other Condensed Matter',
        'quant-gas':'Quantum Gases',
        'soft':'Soft Condensed Matter',
        'stat-mech':'Statistical Mechanics',
        'str-el':'Strongly Correlated Electrons',
        'supr-con':'Superconductivity',
        'gr-qc':'General Relativity and Quantum Cosmology',
        'hep-ex':'High Energy Physics - Experiment',
        'hep-lat':'High Energy Physics - Lattice',
        'hep-ph':'High Energy Physics - Phenomenology',
        'hep-th':'High Energy Physics - Theory',
        'math-ph':'Mathematical Physics',
        'AO':'Adaptation and Self-Organizing Systems',
        'CD':'Chaotic Dynamics',
        'CG':'Cellular Automata and Lattice Gases',
        'PS':'Pattern Formation and Solitons',
        'SI':'Exactly Solvable and Integrable Systems',
        'nucl-ex':'Nuclear Experiment',
        'nucl-th':'Nuclear Theory',
        'acc-ph':'Accelerator Physics',
        'ao-ph':'Atmospheric and Oceanic Physics',
        'app-ph':'Applied Physics',
        'atm-clus':'Atomic and Molecular Clusters',
        'atom-ph':'Atomic Physics',
        'bio-ph':'Biological Physics',
        'chem-ph':'Chemical Physics',
        'class-ph':'Classical Physics',
        'comp-ph':'Computational Physics',
        'data-an':'Data Analysis, Statistics and Probability',
        'ed-ph':'Physics Education',
        'flu-dyn':'Fluid Dynamics',
        'gen-ph':'General Physics',
        'geo-ph':'Geophysics',
        'hist-ph':'History and Philosophy of Physics',
        'ins-det':'Instrumentation and Detectors',
        'med-ph':'Medical Physics',
        'optics':'Optics',
        'plasm-ph':'Plasma Physics',
        'pop-ph':'Popular Physics',
        'soc-ph':'Physics and Society',
        'space-ph':'Space Physics',
        'quant-ph':'Quantum Physics',
    },
    'Quantitative Biology':{
        'q-bio':'Quantitative Biology',
        'BM':'Biomolecules',
        'CB':'Cell Behavior',
        'GN':'Genomics',
        'MN':'Molecular Networks',
        'NC':'Neurons and Cognition',
        'OT':'Other Quantitative Biology',
        'PE':'Populations and Evolution',
        'QM':'Quantitative Methods',
        'SC':'Subcellular Processes',
        'TO':'Tissues and Organs',
    },
    'Quantitative Finance':{
        'q-fin':'Quantitative Finance',
        'CP':'Computational Finance',
        'EC':'Economics',
        'GN':'General Finance',
        'MF':'Mathematical Finance',
        'PM':'Portfolio Management',
        'PR':'Pricing of Securities',
        'RM':'Risk Management',
        'ST':'Statistical Finance',
        'TR':'Trading and Market Microstructure',
    },
    'Statistics':{
        'stat':'Statistics',
        'AP':'Applications',
        'CO':'Computation',
        'ME':'Methodology',
        'ML':'Machine Learning',
        'OT':'Other Statistics',
        'TH':'Statistics Theory',
    },

}

full_name_to_subjects = defaultdict(str)

# 遍历原始数据
for subject, abbreviations in direction_label.items():
    for full_name in abbreviations.values():  # 直接取全称
        full_name_to_subjects[full_name]=subject
# print(full_name_to_subjects)
# exit()
def data_index(key):
    start_date = datetime(2007, 5, 1)
    # 获取当前时间
    now = datetime.now()
    # 获取当前时间上个月的月份
    if now.month == 1:
        last_month_year = now.year - 1
        last_month = 12
    else:
        last_month_year = now.year
        last_month = now.month - 1
    # 获取上一个月的最后一天
    now = datetime(last_month_year, last_month, calendar.monthrange(last_month_year, last_month)[1])
    print(now)
    data = {}
    while start_date <= now:
        # 将日期格式化为 'YYYY-MM'
        data[start_date.strftime('%Y-%m')]=0
        # 增加一个月
        start_date = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1)
    zero_data = deepcopy(data)
    #方向对应的数据
    file = '/mnt/data2/jingxz/system/data_message/data.jsonl'
    author_dict = defaultdict(int)
    #方向对应的学科数据
    categories = {}
    categories_sum = defaultdict(int)
    with open(file,'r') as f:
        key = key.lower()
        for line in tqdm(f):
            item = json.loads(line)
            if key in item.get('title') or key in item.get('abstract'):
                #相关关键词的论文的数量
                time = datetime.strptime(item['update_date'],'%Y-%m-%d')
                time = time.strftime('%Y-%m')
                if time not in data:
                    continue
                data[time] += 1
                for category in item.get('categories'):
                    if category not in categories:
                        categories[category]=deepcopy(zero_data)
                        categories[category][time]+=1
                        categories_sum[category]+=1
                    else:
                        categories[category][time]+=1
                        categories_sum[category]+=1
                for name in item['authors']:
                    author_dict[name] += 1
    # print(categories_sum)
    all_category ={}
    for i in range(1,2):
        for name in categories_sum:
            if name in full_name_to_subjects:
                if full_name_to_subjects[name] in all_category:
                    all_category[full_name_to_subjects[name]]['child'][name]=categories_sum[name]
            else:
                all_category[name]={'sum':categories_sum[name],'child':{}}
    all_category_result={
        'id':key,
        'children':[]
    }

    for sub in all_category:
        child = []
        for dir in all_category[sub]['child']:
            child.append({'id':f"{dir}: {all_category[sub]['child'][dir]}"})
        subject={'id':f"{sub}: {all_category[sub]['sum']}", 'children': child}
        all_category_result['children'].append(subject)

    #获得作者关系信息数据
    sorted_authors = sorted(
        author_dict.items(),  # 获取 (key, value) 对
        key=lambda item: item[1],  # 按值排序
        reverse=True  # 降序（从大到小）
    )[:10]  # 取前 50 
    sorted_authors = OrderedDict(sorted_authors)
    print(sorted_authors)
    top_20_link_author = defaultdict(lambda:set())
    with open(file,'r') as f:
        key = key.lower()
        for line in tqdm(f):
            item = json.loads(line)
            if key in item.get('title') or key in item.get('abstract'):
                #相关关键词的论文的数量
                for name in item['authors']:
                    if name in sorted_authors:
                        for other_name in item['authors']:
                            if other_name in sorted_authors and other_name != name:
                                top_20_link_author[name].add(other_name)
    print(top_20_link_author)
    #计算边的数目
    distance_1_pairs=[]
    all_authors = set(sorted_authors.keys())
    all_authors = sorted(all_authors)
    author_to_index = {author: idx for idx, author in enumerate(all_authors)}
    n = len(all_authors)
    adj_matrix = np.zeros((n, n), dtype=int)  # 初始化全 0 矩阵
    for author, collaborators in top_20_link_author.items():
        i = author_to_index[author]
        for collaborator in collaborators:
            j = author_to_index[collaborator]
            adj_matrix[i][j] = 1  # 合作记为 1
            adj_matrix[j][i] = 1  # 无向图，对称矩阵
            if i<j:
                distance_1_pairs.append({'source':i, 'target':j})

    # 查看邻接矩阵片段

    #计算距离为2且不连通的作者边
    A_squared = np.dot(adj_matrix, adj_matrix)
    distance_2_pairs = []
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # 避免重复检查 i<j
            if adj_matrix[i][j] == 0 and A_squared[i][j] > 0 and i<j:  # 直接不相连但距离为 2
                distance_2_pairs.append({'source': i, 'target':j})
    points = []
    for name in author_to_index:
        points.append({'id':author_to_index[name], 'name':name})
    # print(points)
    # print(distance_1_pairs)
    # print(distance_2_pairs)
    author_link_data = {'points':points,
                        'distance_1_pairs':distance_1_pairs,
                        'distance_2_pairs':distance_2_pairs}

    all_data = {'monthes': list(data.keys()), 'papers': list(data.values())}
    for key_name, values in categories.items():
        all_data[key_name] = [values.get(month, 0) for month in all_data['monthes']]
    df_papers = pd.DataFrame(all_data)
    # print(df_papers)
    #论文的数量
    output_file = f'/mnt/data2/jingxz/system/data_message/store_keys/{key}_papers_counts.xlsx'
    # print(output_file)
    with pd.ExcelWriter(output_file) as writer:
        df_papers.to_excel(writer, sheet_name='Sheet1', index=False)
    
    #作者的数量
    author_data = {'author_dict':author_dict,
                   'author_link_data':author_link_data,
                   'category':all_category_result}
    author_file = f'/mnt/data2/jingxz/system/data_message/store_keys/{key}_authors_counts.json'
    with open(author_file, "w", encoding="utf-8") as f:
        json.dump(author_data, f, ensure_ascii=False, indent=None)
        

def data_predict(key):
    data_path = f'/mnt/data2/jingxz/system/data_message/store_keys/{key}_papers_counts.xlsx'
    command = [
        sys.executable,  # 使用当前Python解释器
        "-u",
        "../MixF/run.py",
        "--random_seed", "2025",
        "--is_training", "0",
        "--data_path", data_path,
        "--data", 'arxiv',
        "--root_path", '/mnt/data2/jingxz/system/',
        "--model_id", key,
        "--model", "MixF",
        "--key", str(key),  # 确保key转换为字符串
        "--do_key", "true",
        "--seq_len", "12",
        "--pred_len", "12",
        "--input_channels_len", "1",
        "--output_channels_len", "1",
        "--d_model", "64",
        "--patch_len", "12",
        "--stride", "8",
        "--gpu", "0",
        "--n_heads", "2",
        "--e_layers", "5",
        "--des", "Exp",
        "--alpha", "0.5",
        "--do_predict",  # 注意：这是一个标志，后面无需值
    ]

    # 执行命令并捕获输出
    result = subprocess.run(command)

    # 检查执行结果
    if result.returncode == 0:
        print("脚本执行成功！输出如下：")
        print(result.stdout)
    else:
        print(f"脚本执行失败，错误码：{result.returncode}")
        print("错误信息：")
        print(result.stderr)
key='ai'
# data_index(key)
# data_predict('ai')


# 
# points: {
    # id: number,
    # name: string,
# }[],
# edges: {
#   form: number, // id
#   target: number, // id
# }
# 