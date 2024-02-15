import pandas as pd
import numpy as np
import requests
import xmltodict
import json, re, os, sys
import asyncio

from bs4 import BeautifulSoup as soup
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime


BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

from utils import get_content, get_hyper_text_str, get_figcaption_str, get_title, req_get, run_scrape

def scrape_old(url_target, n_pages, is_reverse, max_days, batch_size, output_path, debug=False):
    indices = list(range(1, n_pages))
    all_data = []
    it = 0
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    debug_max_doc = 10
    
    if is_reverse:
        indices.reverse()
    
    for page in indices:
        url_target = url_target.format(page)
        r = req_get(url_target)
        data = xmltodict.parse(r.content)
    
        data_link = data['urlset']['url']
        source = url_target.split("/")[2]
    
        for target in tqdm(data_link, desc=f"page-{page}"):
            doc_url = target['loc']
            doc_date = target['lastmod']
            age = today - pd.to_datetime(doc_date[:10])
            age = age.days
            if age > max_days:
                continue

            if "bbc" in source:
                if not (
                    ("https://www.bbc.com/news/" in doc_url)
                    or ("https://www.bbc.com/newsround/" in doc_url)
                    or ("https://www.bbc.com/sport/" in doc_url)
                    or ("https://www.bbc.com/weather/" in doc_url)
                ):
                    continue


            r_doc = req_get(doc_url)
            soup_data = soup(r_doc.text, 'html.parser')
            
            try:
                doc_title = get_title(soup_data)
            except:
                continue
            content = get_content(soup_data)
            hyper_text_str = get_hyper_text_str(soup_data)
            figcaption_text = get_figcaption_str(soup_data)
    
            if not doc_date:
                continue
            
            all_data.append(
                {
                    "source": source,
                    "title": doc_title,
                    "date": doc_date,
                    "content": content,
                    "hypertext": hyper_text_str,
                    "figcaption": figcaption_text,
                    "url": doc_url,
                }
            )
            
            if debug:
                debug_max_doc -= 1
                batch_size = 5
    
                if debug_max_doc <= 0:
                    debug_max_doc = 10
                    break
    
            if len(all_data) >= batch_size:
                data_fin = pd.DataFrame(all_data)
                output_path_file = output_path.format(today_str, it)

                data_fin.to_csv(
                    output_path_file,
                    mode="a",
                    header=not os.path.exists(output_path_file),
                    index=False,
                )
                all_data = []
                it += 1
                if debug:
                    exit()
                    
        if age > max_days:
            continue

def filter_data_link(data_links, source, today, max_days):
    if "bbc" in source:
        data_link = []
        for l in data_links:
            doc_url = l['loc']
            if not (
                ("https://www.bbc.com/news/" in doc_url)
                or ("https://www.bbc.com/newsround/" in doc_url)
                or ("https://www.bbc.com/sport/" in doc_url)
                or ("https://www.bbc.com/weather/" in doc_url)
            ):
                continue
            else:
                data_link.append(l)
    else:
        data_link = data_links

    data_link_clean = []

    max_age = 0
    
    for l in data_link:
        date_publish_str = l.get('lastmod', "2000-01-01")[:10]
        date_publish = datetime.strptime(date_publish_str, "%Y-%m-%d")

        age = today - date_publish
        age = age.days
        max_age = max(max_age, age)
        if age > max_days:
            continue

        data_link_clean.append(
            (l['loc'], l['lastmod'][:10], source)
        )
    
    return data_link_clean, max_age

async def scrape_old_async(url_target, n_pages, n_workers, is_reverse, max_days, batch_size, output_path, debug=False):
    indices = list(range(1, n_pages))
    all_data = []
    it = 0
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    debug_max_doc = 10
    
    if is_reverse:
        indices.reverse()
    
    for page in indices:
        url_target = url_target.format(page)
        r = req_get(url_target)
        data = xmltodict.parse(r.content)
    
        data_links = data['urlset']['url']
        source = url_target.split("/")[2]
        
        data_link, max_age = filter_data_link(data_links, source, today, max_days)


        data_link_chunks = np.array_split(data_link, len(data_link)//n_workers)

        for pkg in tqdm(data_link_chunks):
            output_data = await run_scrape(pkg)
            all_data.extend(output_data)
            
            if debug:
                debug_max_doc -= 1
                batch_size = 5
    
                if debug_max_doc <= 0:
                    debug_max_doc = 10
                    break
    
            if len(all_data) >= batch_size:
                data_fin = pd.DataFrame(all_data)
                output_path_file = output_path.format(today_str, it)

                data_fin.to_csv(
                    output_path_file,
                    mode="a",
                    header=not os.path.exists(output_path_file),
                    index=False,
                )
                all_data = []
                it += 1
                if debug:
                    exit()
                    
        if max_age > max_days:
            continue