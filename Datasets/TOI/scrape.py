from newspaper import Article
import csv
import os.path

filedir="REGION/{}.csv"

def scrape_url(url):
    article = Article(url, language="en")
    article.download() 
    article.parse() 
    article.nlp()
    return {
        "Title" : article.title,
        "Text" : article.text,
        "Keywords" : article.keywords,
        "Summary" : article.summary }

def append(region,dictionary):
    csv_path=filedir.format(region)
    print(csv_path)
    if not os.path.exists(csv_path):
        with open(csv_path, "w+") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
            writer.writeheader()
    with open(csv_path, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        writer.writerow(dictionary)