import requests
import json
import pandas as pd
import os
import numpy as np
from tqdm.asyncio import tqdm
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
import threading
import asyncio
import aiohttp
import aiofiles


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

async def get_text(data, url, headers):
    max_retries = 10
    retries = 0
    async with aiohttp.ClientSession() as session:
        while retries < max_retries:
            try:
                async with session.post(url, headers=headers, json=data, timeout=10000) as response:
                    response.raise_for_status()
                    if response.status == 200:
                        return await response.json()
                    else:
                        retries+=1
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"请求失败，正在重试 ({retries + 1}/{max_retries}): {e}")
                retries += 1
        print("达到最大重试次数，请求失败。")
    return None

async def get_text_fromgpt(option, data):
    # print(data.shape, data[0].shape, data[1].shape)
    paper_num = ' '.join([str(x) for x in data[0]])
    author_num = ' '.join([str(x) for x in data[1]])
    # print(paper_num)
    # print(author_num)
    api_key = _require_env("GPTGOD_API_KEY")
    url = os.getenv("GPTGOD_API_URL", "https://gptgod.cloud/v1/chat/completions")
    model_name = os.getenv("GPTGOD_MODEL", "grok-3-deepersearch-r")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    content = f"""我现在通过我的模型将学科方向中关于{option}在arxiv数据集上,关于该方向从2007年5月到最新的时间每个月的该学科的论文数量和作者数量的信息,同时我通过我得算法将数据也进行了对未来十二个月的数据的预测。
    该方向的论文数量信息为:{paper_num}。
    该方向的作者数量信息为:{author_num}。
    我现在需要你完成以下几个任务：
    1、描述过去这个学科方向的大致情况,通过我给你的数据信息对我预测的情况进行大概的情况分析。在分析预测效果的时候，需要将过去的数据进行大致的情况进行描述，以及对未来的预测分析进行详细的分析。
    2、该学科方向在过去时间中举例出引用量较高或者对该学科方向具有极大重大影响的论文十篇论文,并给出论文的名字和链接。
    """
    data = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": "你是一个十分厉害并且在通晓各个学科方向的信息和数据,现在你需要将分析用户提出的相关信息,然后将数据信息进行分析，最终将得到的信息进行生成，最终返回给用户。"},
            {"role": "user", "content": content}
        ]
    }
    # print('already send meesage')
    print(option)
    response = await get_text(url=url, headers=headers, data=data)
    # print('already accept message'
    if response is not None:
        if response['choices'][0]['finish_reason'] == 'stop':
            return response['choices'][0]['message']['content']
        else:
            return "ERROR"
    else:
        return "ERROR"
    



async def process_samples(samples, path, pbar):
    for sample in samples:
        name = sample['name']
        data = sample['data']
        text = await get_text_fromgpt(name, data)
        pbar.update(1)
        if text != "ERROR":
            async with aiofiles.open(path, mode='a') as f:
                await f.write(json.dumps({'name':name,'text':text})+'\n')
        



def distribute_samples(samples, n_processes):
    chunk_size = len(samples)//n_processes +1
    return [samples[i * chunk_size : (i + 1) * chunk_size] for i in range(n_processes)]

async def main(samples, path, n_processes=5):
    with tqdm(total=len(samples), desc="subject analysis", ncols=80) as pbar:
        process_samples_list = distribute_samples(samples, n_processes)
        tasks = [ asyncio.create_task(process_samples(process_samples_list[i], path, pbar)) for i in range(n_processes)]
        await asyncio.gather(*tasks)






data_path = './data_message/predicts.xlsx'
df_row = pd.read_excel(data_path)
cols = list(df_row.columns)
cols.remove('monthes')
show_options = set()

path = './data_message/analysis_data.jsonl'
if os.path.exists(path):
    with open(path,mode='r') as f:
        for line in f:
            data = json.loads(line)
            show_options.add(data.get('name'))
else:
    with open(path, mode='w', encoding='utf-8') as ff:
        print("文件创建成功！")

samples=[]

for i in tqdm(range(0,len(cols),2)):
    data_papers = cols[i]
    data_authors = cols[i+1]
    if '_papers' in data_papers:
        name = data_papers.removesuffix('_papers')
        if name  not in show_options:
            data = np.array([df_row[data_papers],df_row[data_authors]])
            samples.append({'name':name, 'data':data})\

asyncio.run(main(samples,path))



