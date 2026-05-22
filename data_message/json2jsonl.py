import copy
# import pickle as pkl
import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
path = "/mnt/data2/jingxz/system/MixF/dataset/arxiv-metadata-oai-snapshot.json"

stroage_file = '/mnt/data2/jingxz/system/data_message/data.jsonl'


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


already_ids=set()
if os.path.exists(stroage_file):
    with open(stroage_file,'r') as f:
         for line in tqdm(f):
            item = json.loads(line)
            already_ids.add(item['id'])



with open(path, 'r') as f:
    contents = f.readlines()
for content in tqdm(contents):
    content = json.loads(content)
    if content['id'] not in already_ids:
        #学科方向进行整合
        categories = set()
        for category in content['categories'].split(' '):
            category_name=category.split('.')
            subject = category_name[0]
            if subject not in subject_label:#如果该分类不存在于官方的分类中，直接删除
                continue
            subject_name = subject_label[subject]
            categories.add(subject_name)
            #部分文章主副标题之分
            if len(category_name)>1:
                direction = category.split('.')[1]
                direction_name = direction_label[subject_name][direction]
                categories.add(direction_name)
        #作者
        authors=[]
        for author in content['authors_parsed']:
            authors.append(author[1]+' '+author[0])
        data = {
            'id':content['id'],
            'title':content['title'].lower(),
            'update_date':content['update_date'],
            'abstract':content['abstract'].lower(),
            'authors':list(authors),
            'categories':list(categories),
        }
        with open(stroage_file, mode='a') as f:
            f.write(json.dumps(data)+'\n')

