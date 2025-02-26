import os
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
#from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import spacy

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
ner_nlp = spacy.load('en_core_web_sm')

knowledgebase = pickle.load(open("knowledgebaseunmod.p", "rb"))
knowledgebasewords = ["nikoli", "puzzle", "sudoku", "rules", "book", "magazine", "japan", "solve", "cells", "block"]

#print(knowledgebase)

usernameunparsed = input("Hello, what is your name?\n>")
username = ""
while username == "":    
    unamedoc = ner_nlp(usernameunparsed)
    names = [x for x in unamedoc.ents if x.label_ == "PERSON"]
    if names:
        #print(names)
        username = names[0].text
    else:
        #Some names/places never get accepted depending on the context.
        #I.e. if the user is named Salem, the NER always misclassifies it as a location, even if phrased as "My name is Salem".
        #After trying to grab it once with NER, it just defaults to taking the full string to prevent a loop.
        username = input("Sorry, I didn't quite get that. What is your name?\n>")

cwd = os.path.dirname(os.path.abspath(__file__))
user_file_path = os.path.join(cwd, "usermodels/" + username + ".p")

if os.path.exists(user_file_path):
    userprofile = pickle.load(open(user_file_path, "rb"))
    print("Oh, hello, " + username + "! I remember you--you're the person from "+ userprofile["hometown"] +" who likes " + userprofile["likes"] + " and dislikes " + userprofile["dislikes"] + ".")

if not os.path.exists(user_file_path):
    hometownunparsed = input("Nice to meet you! What is your hometown?\n>")
    hometown = ""
    while hometown == "":    
        placedoc = ner_nlp(hometownunparsed)
        places = [x for x in placedoc.ents if x.label_ == "GPE"]
        if places:
            #print(places)
            hometown = places[0].text
        else:
            #Some names/places never get accepted depending on the context.
            #After trying to grab it once with NER, it just defaults to taking the full string.
            hometown = input("Sorry, I didn't quite get that. Where are you from?\n>")

    liked = input("Alright! What is something you like?\n>")
    disliked = input("Okay, what is something you dislike?\n>")
    userprofile = {"hometown": hometown,
                   "likes": liked,
                   "dislikes": disliked}
    
    pickle.dump(userprofile, open(user_file_path, "wb"))
    print("Nice to meet you, " + username + "! I am a chatbot with information on the Japanese puzzle publisher, Nikoli, which is most well known for creating Sudoku.")


reply = input("What would you like to talk about? (type \"exit\" to quit.)\n>")
while "exit" not in reply:
    foundwords=[]
    for keyword in knowledgebasewords:
        if keyword.lower() in reply.lower():
            foundwords.append(keyword)
    
    #print(foundwords)

    replywords = [wnl.lemmatize(t.lower()) for t in word_tokenize(reply) if t.isalpha() and not t in stop_words]
    bestsentence = "Sorry, I don't understand."
    bestcosine = -1

    if foundwords != []:
        foundwords.reverse()
        sentences = knowledgebase[foundwords[0]]
        random.shuffle(sentences)

        #Finds the sentence with the highest cosine similarity

        for sentence in sentences:
            words = [wnl.lemmatize(t.lower()) for t in word_tokenize(sentence) if t.isalpha() and not t in stop_words]
            countcommonwords = 0
            for word in words: 
                if word in replywords:
                    countcommonwords = countcommonwords + 1
            cosinesim = countcommonwords/(np.sqrt(len(words))*np.sqrt(len(replywords)))
            if cosinesim > bestcosine:
                bestcosine = cosinesim
                bestsentence = sentence
        print(bestsentence)
    else:
        print('Sorry, I don\'t understand. I can talk about the company Nikoli, their puzzles (including Sudoku), their rules and how to solve them, Nikoli\'s books and magazines, or common puzzle elements like cells and blocks.')

    #print(foundwords)
    reply = input(">")
    


    
    ''' Scrapped sentiment analysis code
    if "yes" in sudokuopinion or "yeah" in sudokuopinion or "yep" in sudokuopinion:
        likesudoku = True
    if "no" in sudokuopinion or "nope" in sudokuopinion or "nah" in sudokuopinion:
        likessudoku = False
    #if can't easily find, sentiment analysis
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(sudokuopinion)
    for k in sorted(scores):
        print('{0}: {1}, '.format(k, scores[k]), end='')
    print(sorted(scores)[1])
    '''


#check if username file exists.
    #if no, then create user file
        #ask Where are you from
        #Ask if they like Sudoku, save
            #Maybe measure using sentiment analysis? Or just do a simple check for "yeah yes yep" and "no nah nope"
        #Ask if they know about Nikoli
        #Well, I am a chatbot with a lot of information about it
        #I can discuss these topics
        #GOTO main loop
    #if yes, then welcome back
        #Oh hello/Ah yes, You're the person from ____ who likes ___ and dislikes ___!
        #Remind what topics can be discussed
        #GOTO main loop

#MAINLOOP
    #I can discuss these topics. You can also reply quit to stop talking.
    #Match input with knowledgebase with some sort of threshold
        #if none match say something like "I'm sorry, I don't understand."
            #if a second time "Sorry, that must be outside my knowledge. I am able to talk about ______"
    #
