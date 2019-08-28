from newspaper import Article

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