import json

from newspaper import Article
from rake_nltk import Rake
file = open("example_1.json").read()
arr = json.loads(file)

all_documents = []
for article in arr:
    if "content" in article and article['source_content'] != 'video' and article['validated'] = -2:

        url = article['_id']
        art = Article(url, language='en')  # English
        try:
            art.download()
            art.parse()
            art_content =  art.text
            print(article['_id'])
            all_documents.append(art_content)
        except:
            print('bad article')
            print(article['source_content'])
            print(article['_id'])
            continue
