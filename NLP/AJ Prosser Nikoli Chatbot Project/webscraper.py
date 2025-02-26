import os
import requests
import math
from bs4 import BeautifulSoup
import re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.text import Text
import numpy
import pickle

num_pages_to_scrape = 30
linksperdomain = 15
linksperpage = 99

stopwords = stopwords.words('english')

#Reading file as raw text
cwd = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cwd, "webscraperinput.txt")
file = open(file_path)
urls = file.read().splitlines()

incrementor = 0
interincrementor = 0
filenames = []
visitedsites = {}
visitedpages = []

print("Webcrawling...")

while incrementor < num_pages_to_scrape:
    url = urls[interincrementor]
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    interincrementor = interincrementor + 1
    #print("Interincrementor: " + str(interincrementor))
    #print("incrementor: " + str(incrementor))

    #Stripping tags we don't want
    for each in ['header','footer']:
        removetags = soup.find_all(each)
        for s in removetags:
            if s is not None:
                s.extract()  
    
    #Unwrapping formatting tags
    for each in ['i','b']:
        removetags = soup.find_all(each)
        for s in removetags:
            if s is not None:
                s.unwrap()  

    filename = "outfilesunprocessed/" + url.replace(".","DOT").replace("https://","").replace("/","SLASH") + ".txt"
    file_path = os.path.join(cwd, filename[:100])

    if not os.path.exists(file_path):
        incrementor = incrementor + 1

    outfile = open(file_path, "w")

    unmodtext = soup.body.get_text("\n", strip=True)

    #foreigntext = re.compile('[\u3040-\u9faf]]') #unicode ranges which include japanese characters

    unmodtext = re.sub(re.compile('[\u3040-\u309F]'),'',unmodtext)
    unmodtext = re.sub(re.compile('[\u30A0-\u30FF]'),'',unmodtext)
    unmodtext = re.sub(re.compile('[\u4300-\u9faf]'),'',unmodtext)

    outfile.write(unmodtext)
    visitedpages.append(url)
    #print("Saved File")
    
    
    linkssofar = 0

    linksonthispage = soup.find_all('a')
    #print("Number of links on this page: " + str(len(linksonthispage)))

    #numpy.random.shuffle(linksonthispage)

    for link in linksonthispage:
        

        if link.get("href") is not None and link.get("href").startswith("https://") and linkssofar < linksperpage:
            lazydomain = link.get("href")[5:15].replace(".","DOT").replace("https://","").replace("/","SLASH")
            
            for seenpage in visitedpages:
                if seenpage == url:
                    bantest = True
                    #print("Already Traversed Page" + url)
        

            excludes = ["wikipedia", "wikimedia", "wikidata", "google", "viaf", "archive.org", ".fr"]

            bantest = False
            gooddomain = False
            url = link.get("href")

            for excludedsite in excludes:
                if excludedsite in url:
                    bantest = True

            if visitedsites.get(lazydomain,0) < linksperdomain and not bantest: #Conditional force traversal to other sites
                visitedsites[lazydomain] = visitedsites.get(lazydomain,0)+1
                urls.append(url)
                #filenames.append(file_path)
                #print("Appending URL: " + link.get("href"))
                linkssofar = linkssofar + 1
    outfile.close()

#print(filenames)

uncleaneddir = os.path.join(cwd, "outfilesunprocessed/")
cleaneddir = os.path.join(cwd, "outfiles/")
filenames = os.listdir(uncleaneddir)
#print(filenames)
doc_nikoli = ""
nikoli_corpus = ""

print("Webcrawling done! Cleaning the files...")

for uncleanfile in filenames:
    inputfile = open(os.path.join(uncleaneddir, uncleanfile), "r")
    unprocessedtext = inputfile.read()
    outputfile = open(os.path.join(cleaneddir, uncleanfile),'w')
    outputfile.write(unprocessedtext)

    doc_nikoli = doc_nikoli + unprocessedtext.lower() #For NLP
    nikoli_corpus = nikoli_corpus + unprocessedtext #For searching to create the knowledge base
    
    
#filter out all lines shorter than 15 chars
nikoli_split = nikoli_corpus.splitlines()
nikoli_corpus = ''
for line in nikoli_split: 
    if len(line) > 15:
        nikoli_corpus = nikoli_corpus + line + "\n"


print("Finding relevant words...")
#TF, adapted from the Keywords with tf-idf notebook by Karen Mazidi.
vocab = set()
def create_tf_dict(doc):
    tf_dict = {}
    tokens = word_tokenize(doc)
    tokens = [w for w in tokens if w.isalpha() and w not in stopwords]
     
    # get term frequencies
    for t in tokens:
        if t in tf_dict:
            tf_dict[t] += 1
        else:
            tf_dict[t] = 1
            
    # get term frequencies in a more Pythonic way
    token_set = set(tokens)
    tf_dict = {t:tokens.count(t) for t in token_set}
    
    # normalize tf by number of tokens
    for t in tf_dict.keys():
        tf_dict[t] = tf_dict[t] / len(tokens)
        
    return tf_dict

#importing of dummy sample documents to compare using TF-IDF
#Anatomy textbook
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'anatsample.txt'), 'r') as f:
    doc_anat = f.read().lower()
    doc_anat = doc_anat.replace('\n', ' ')
    
#Economics textbook
with open(os.path.join(cwd ,'econsample.txt'), 'r') as f:
    doc_econ = f.read().lower()
    doc_econ = doc_econ.replace('\n', ' ')

#Text of Moby Dick
doc_moby = gutenberg.raw("melville-moby_dick.txt")[:15000]


print("...Computing term frequencies...")
#We are comparing our dataset to other unrelated datasets.
#Since TF-IDF is a method of comparison, this is needed to find the words most unique to our dataset.
print("   ...Nikoli...")
tf_nikoli = create_tf_dict(doc_nikoli)
print("   ...Test-data 1...")
tf_anat = create_tf_dict(doc_anat)
print("   ...Test-data 2...")
tf_econ = create_tf_dict(doc_econ)
print("   ...Test-data 3...")
tf_moby = create_tf_dict(doc_moby)
num_docs = 4

# add to vocab
vocab = set(tf_nikoli.keys())
vocab = vocab.union(set(tf_anat.keys()))
vocab = vocab.union(set(tf_econ.keys()))
vocab = vocab.union(set(tf_moby.keys()))


print("...Computing IDF...")
#IDF
idf_dict = {}

vocab_by_topic = [tf_nikoli.keys(), tf_anat.keys(), 
                  tf_econ.keys(), tf_moby.keys()]

for term in vocab:
    temp = ['x' for voc in vocab_by_topic if term in voc]
    idf_dict[term] = math.log((1+num_docs) / (1+len(temp))) 

#TF-IDF
def create_tfidf(tf, idf):
    tf_idf = {}
    for t in tf.keys():
        tf_idf[t] = tf[t] * idf[t] 
        
    return tf_idf

tf_idf_nikoli = create_tfidf(tf_nikoli, idf_dict)

doc_term_weights = sorted(tf_idf_nikoli.items(), key=lambda x:x[1], reverse=True)
print("\nNikoli Relevant Words: ", doc_term_weights[:40])

#Prompt the user to search:

sent_t_nikoli = nltk.sent_tokenize(nikoli_corpus)
endflag = False
''' Section that allows the user to search up sentences containing a word. Used to determine the best critical words.
while not endflag:
    searchword = input("What word would you like to search the corpus for?: ")

    if searchword == "exit":
        endflag = True

    sents = []
    for sentence in sent_t_nikoli:
        if searchword in sentence and len(sentence) > len(searchword) + 10:
            if len(sentence) < 400:
                sents.append(sentence.replace('\n', ' '))
    
    numpy.random.shuffle(sents)

    if len(sents) < 20:
        print(*sents, sep='\n---\n')
    else:
        print(*sents[:15], sep='\n---\n')
'''
knowledgebasewords = ["nikoli", "puzzle", "sudoku", "rules", "book", "magazine", "Japan", "solve", "cells", "block"]
knowledgebasedict = {}

for kbword in knowledgebasewords:
    sents = []
    for sentence in sent_t_nikoli:
        if kbword in sentence and len(sentence) > len(kbword) + 10:
            if len(sentence) < 400:
                sents.append(sentence.replace('\n', ' '))
    numpy.random.shuffle(sents)

    knowledgebasedict[kbword] = sents

pickle.dump(knowledgebasedict, open("knowledgebaseunmod.p", "wb"))
print("Knowledge base saved!")

#print(knowledgebasedict)
#wnl = WordNetLemmatizer()
    #stop_words = set(stopwords.words('english'))
    #words = [t.lower() for t in word_tokenize(unprocessedtext) if t.isalpha() and not t in stop_words]

#print(urls)
