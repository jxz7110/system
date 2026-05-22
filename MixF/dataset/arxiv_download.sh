conda activate torch
rm arxiv.zip
kaggle datasets download Cornell-University/arxiv
unzip -o arxiv.zip
conda activate torch