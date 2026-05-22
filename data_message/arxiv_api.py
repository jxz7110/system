import arxiv
import time
from tqdm import tqdm


def arxiv_find_key_paper(key):
    # Construct the default API client.
    client = arxiv.Client()

    # Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
    query = key,
    max_results = 10,
    sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)

    # print('result length:', len(client.results(search)))
    i=1
    papers=[]
    text ="我们将根据您提供最新和关键词相关的十篇论文。\n\n"
    for result in results:
        # print(result.entry_id, result.title, result.authors, result.summary, result.pdf_url)
        # 1. **\"A Higgs Boson at 125 GeV and Enhancement in the Diphoton Decay Channel\"** (2012)\n   - 引用量: 5000+\n   - 链接: https://arxiv.org/abs/1207.1347\n   - 重要性: 希格斯玻色子发现后的首批理论解释之一\n\n
        id = result.entry_id
        title = result.title
        authors = result.authors
        author_text=""
        for author in authors:
            author_text=author_text+str(author)+', '
        summary = result.summary.replace('\n','')
        pdf_url = result.pdf_url
        data=f'## {i}.**\"{title}\"** \n - Authors: {author_text}\n - Abstract: {summary}\n - pdf_url: {pdf_url}\n\n'
        text = text + data
        i+=1
    return text
# `results` is a generator; you can iterate over its elements one by one...

# ...or exhaust it into a list. Careful: this is slow for large results sets.
# all_results = list(results)
# print([r.title for r in all_results])

# For advanced query syntax documentation, see the arXiv API User Manual:
# https://arxiv.org/help/api/user-manual#query_details
# search = arxiv.Search(query = "au:del_maestro AND ti:checkerboard")
# first_result = next(client.results(search))
# print(first_result)

# Search for the paper with ID "1605.08386v1"
# search_by_id = arxiv.Search(id_list=["1605.08386v1"])
# # Reuse client to fetch the paper, then print its title.
# first_result = next(client.results(search))
# print(first_result.title)
# import urllib, urllib.request
# import feedparser
# from datetime import datetime, timedelta

# # 获取当前时间
# current_time = datetime.now()
# now_time = current_time.strftime("%Y%m%d0000")
# # 计算四年前的时间
# time_four_years_ago = current_time.replace(year=current_time.year - 4)
# time_four_years_ago = time_four_years_ago.strftime("%Y%m%d0000")

# print("四年前的时间是:", time_four_years_ago)
# print("当前时间为:", now_time) 

# # 设置API密钥
# # api_key = 'YOUR_API_KEY'
# # 发送GET请求
# search_query = "AI"
# base_url = "http://export.arxiv.org/api/query?"
# encoded_query = urllib.parse.quote_plus(search_query)
# url = f'{base_url}search_query={encoded_query}+AND+submittedDate:[{time_four_years_ago}+TO+{now_time}]&max_results=1000&sortBy=lastUpdatedDate&start=100'
# print(url)
# feed = feedparser.parse(url)
# i=0
# for entry in feed.entries:
#     i+=1
#     # print(f"标题: {entry.title}")
#     # print(f"作者: {', '.join([author.name for author in entry.authors])}")
#     # print(f"摘要: {entry.summary}")
#     # print(f"链接: {entry.link}")
#     # print("=" * 80)
# print(i)
# response = urllib.request.urlopen(url)

# print(response.read().decode('utf-8'))


# # 打印论文标题
# for paper in data['data']:
#     print(paper['title'])