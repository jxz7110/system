import copy
# import pickle as pkl
import json
import os
from collections import defaultdict
import dill as pkl
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from tqdm import tqdm
path = ".."
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
with open(os.path.join(path, "dataset", "arxiv-metadata-oai-snapshot.json"), 'r') as f:
    contents = f.readlines()

dir_type = {
    'author': set(),
    'paper': 0,
}
dir_num ={
    'author': 0,
    'paper': 0,
}
categories = defaultdict(lambda: defaultdict(lambda: copy.deepcopy(dir_type))) #name : month : dir_type
for content in tqdm(contents):
    content = json.loads(content)
    # print(content)
    #时间计数
    time = datetime.strptime(content['update_date'],'%Y-%m-%d')
    time = time.strftime('%Y-%m')
    #    print(time)
    for category in content['categories'].split(' '):
        category_name=category.split('.')
        subject = category_name[0]
        if subject not in subject_label:#如果该分类不存在于官方的分类中，直接删除
            break
        subject_name = subject_label[subject]
        # print(subject_name)
        #部分文章主副标题之分
        if len(category_name)>1:
            direction = category.split('.')[1]
            direction_name = direction_label[subject_name][direction]
            categories[subject_name][time]['paper']+=1
            categories[direction_name][time]['paper']+=1
            #将每个方向加入作者
            for author in content['authors'].replace('\n','').split(','):
                author=author.strip()
                categories[subject_name][time]['author'].add(author)
                categories[direction_name][time]['author'].add(author)
        else:
            direction_name = direction_label[subject_name][subject]
            categories[direction_name][time]['paper']+=1
            for author in content['authors'].replace('\n','').split(','):
                author=author.strip()
                categories[direction_name][time]['author'].add(author)
categories_time_num=defaultdict(lambda: defaultdict(lambda: copy.deepcopy(dir_num)))
# for name in categories.keys():
#     for month in categories[name].keys():
#         categories_time_num[name][month]['paper']=categories[name][month]['paper']
#         categories_time_num[name][month]['author']=len(categories[name][month]['author'])
# print(categories['computer science'])
# try:
#     existing_df = pd.read_excel("categories.xlsx", sheet_name="Category_Data")
# except FileNotFoundError:
existing_df = pd.DataFrame({'monthes':[]})

for name in categories.keys():
    monthes=[]
    for month in categories[name].keys():
        monthes.append([month,categories[name][month]['paper'],len(categories[name][month]['author'])])
        # papers.append(categories[name][month]['paper'])
        # authors.append(len(categories[name][month]['author']))
    monthes=np.array(monthes)
    print(name)
    sorted_monthes = monthes[np.argsort(monthes[:, 0])]
    # print(sorted_monthes)
    new_df = pd.DataFrame({'monthes':sorted_monthes[:-1,0], name+'_papers':sorted_monthes[:-1,1],name+'_authors':sorted_monthes[:-1,2]})
    existing_df=pd.merge(existing_df, new_df, on='monthes', how='outer')
    existing_df.fillna(value=0,inplace=True)
with pd.ExcelWriter("categories.xlsx", engine='openpyxl', mode='w') as writer:  # 使用 'w' 模式重写文件
    existing_df.to_excel(writer, sheet_name='Category_Data', index=False)

