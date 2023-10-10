from bs4 import BeautifulSoup
import requests, json, lxml

def gsearch(query : str, max: int = 10, country = "us", lang = "en"):
    # https://docs.python-requests.org/en/master/user/quickstart/#passing-parameters-in-urls
        params = {
            "q": query.replace("\"",""), # query example
            "hl": lang,          # language
            "gl": country,          # country of the search, UK -> United Kingdom
            "start": 0,          # number page by default up to 0
            "num": max          # parameter defines the maximum number of results to return.
        }

        # https://docs.python-requests.org/en/master/user/quickstart/#custom-headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        }

        page_limit = 10
        page_num = 0
        data = []

        while True:
            page_num += 1
            html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
            soup = BeautifulSoup(html.text, 'lxml')
            
            for result in soup.select('.tF2Cxc'):
                title = result.select_one('.DKV0Md').text
                link = result.select_one('.yuRUbf a')['href']

                # sometimes there's no description and we need to handle this exception
                try: 
                    snippet = result.select_one('#rso .lyLwlc').text
                except: 
                    snippet = None

                if (link.startswith("http")):
                    data.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet
                    })
                
            if page_num == page_limit:
                break
            if soup.select_one(".d6cvqb a[id=pnnext]"):
                params["start"] += 10
            else:
                break
        
        return data
    

# -------------
# '''
# [
#   {
#     "title": "Tesla: Electric Cars, Solar & Clean Energy",
#     "link": "https://www.tesla.com/",
#     "snippet": "Tesla is accelerating the world's transition to sustainable energy with electric cars, solar and integrated renewable energy solutions for homes and ..."
#   },
#   {
#     "title": "Tesla, Inc. - Wikipedia",
#     "link": "https://en.wikipedia.org/wiki/Tesla,_Inc.",
#     "snippet": "Tesla, Inc. is an American electric vehicle and clean energy company based in Palo Alto, California, United States. Tesla designs and manufactures electric ..."
#   },
#   {
#     "title": "Nikola Tesla - Wikipedia",
#     "link": "https://en.wikipedia.org/wiki/Nikola_Tesla",
#     "snippet": "Nikola Tesla was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist best known for his contributions to the design of the ..."
#   }
# ]
# '''