import sys
import os
import pickle
from argparse import ArgumentParser
from platform import system
from subprocess import Popen
from sys import argv
from sys import stderr
import re
import math
from textblob import TextBlob
import nltk
import operator
from textblob import Word


# In[2]:


stopwords = ["a", "share", "linkthese", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any","", "are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves", "this"]
data = "In 2007 , the country with the highest estimated incidence rate of TB was Swaziland , with 1,200 cases per 100,000 people . India had the largest total incidence , with an estimated 2.0 million new cases . In developed countries , tuberculosis is less common and is found mainly in urban areas . Rates per 100,000 people in different areas of the world were : globally 178 , Africa 332 , the Americas 36 , Eastern Mediterranean 173 , Europe 63 , Southeast Asia 278 , and Western Pacific 139 in 2010 . In Canada and Australia , tuberculosis is many times more common among the aboriginal peoples , especially in remote areas . In the United States Native Americans have a fivefold greater mortality from TB , and racial and ethnic minorities accounted for 84 % of all reported TB cases . "
data2 = "Piano is an Indian classical instrument . Akhil plays Piano and Guitar . Pasta is a South Indian dish made of rice and tamarind . Monica writes poems . Osmosis is the movement of a solvent across a semipermeable membrane toward a higher concentration of solute . In biological systems, the solvent is typically water, but osmosis can occur in other liquids , supercritical liquids, and even gases . When a cell is submerged in water, the water molecules pass through the cell membrane from an area of low solute concentration to high solute concentration . For example, if the cell is submerged in saltwater, water molecules move out of the cell . If a cell is submerged in freshwater, water molecules move into the cell . Russia is divided into eight steps, the first is Yama -- non - killing, truthfulness, non - stealing, continence, and non - receiving of any gifts . Next is Niyama -- cleanliness, contentment, austerity, study, and self - surrender to God . "
data1 = "A revolution in 1332 resulted in a broad-based city government with participation of the guilds , and Strasbourg declared itself a free republic . The deadly bubonic plague of 1348 was followed on 14 February 1349 by one of the first and worst pogroms in pre-modern history : over a thousand Jews were publicly burnt to death , with the remainder of the Jewish population being expelled from the city . Until the end of the 18th century , Jews were forbidden to remain in town after 10 pm . The time to leave the city was signalled by a municipal herald blowing the Grüselhorn ( see below , Museums , Musée historique ) ; . A special tax , the Pflastergeld ( pavement money ) , was furthermore to be paid for any horse that a Jew would ride or bring into the city while allowed to . "
data3 = "Michael Jackson is crying because of injury . "
data5 = "Google bought IBM for 10 dollars . Mike was happy about this deal . Rahil's birthday is on 17th Dec 1995 . Rahil will pay 200 dollars to John at 7:00 PM . Nikita has 10 chocolates . "

# In[3]:


# Preprocessing
# splitted corpus with full stop
# created list of sentences


def pre_process(raw_data):
    for c in [',','!',';','?']:
        raw_data = raw_data.replace(c,'')
    raw_data = raw_data.replace(" %","%")

    #sentences = raw_data.lower().split(" . ")
    sentences = raw_data.split(" . ")
    processed_sentences = []
    for sent in sentences:
        if len(sent)!=0:
            sent = list(sent.split())
            
            for s in sent:
                if s not in stopwords:
                    vocab[s] = vocab.get(s,0) + 1
            processed_sentences.append(sent)

        
    return processed_sentences


# In[4]:


#populating dictionaries which will be used to formulate TF-IDF

def inverse_dict():
    global psent,words_sents_map,total_words_in_sent
#     print(psent)
    for i in range(0,len(psent)):
        temp = {}
        count = 0
        for word in psent[i]:
            if word not in temp:
                temp[word] = 1
            else:
                temp[word] += 1
                
            if word not  in stopwords:
                count += 1
       
        total_words_in_sent[i] = count
        words_sents_map[i] = temp
        


# In[5]:


#Calculating TF-IDF against sentences


def calc_tf_idf():
    global psent,vocab,words_sents_map,total_words_in_sent
    global tfidf
    
    for i in range(len(psent)):
        for word in psent[i]:
            if(word not in stopwords):        
                #print(words_sents_map[i][word],total_words_in_sent[i])

                tf = (words_sents_map[i][word]*1.0)/total_words_in_sent[i]
                #print(tf)
                deno = 0
                for j in range(len(psent)):
                    if(j in words_sents_map and word in words_sents_map[j] and words_sents_map[j][word]!=0):
                        deno += 1
                        
                idf = vocab[word]/deno
                # print("idf",idf)
                tf_idf[word] = tf * idf
    return


# In[6]:


#question generation
# Simple rule based approach
# as of now it generates questions starting with "What".


def genQuestion(sentence,ner):
    
    #print("ner: ",ner)
    """
    outputs question from the given text
    """

    time_flag = 0
    word_ner_map = {}
    for i in range(len(ner)):
        word_ner_map[ner[i][0]] = ner[i][1]
        if ner[i][1] == "TIME" or ner[i][1] == "DATE":
            time_flag = 1

    #print(word_ner_map)    

    if type(sentence) is str:   
        line = TextBlob(sentence) 

    bucket = {}               # Create an empty dictionary


    for i,j in enumerate(line.tags): 
        #print(j)
        if j[1] not in bucket:
            bucket[j[1]] = i  
    
    
    #print(bucket)
    question = '' 
    questions = []           

    l1 = ['NNP', 'VBG', 'VBZ', 'IN']
    l2 = ['NNP', 'VBG', 'VBZ']
    

    l3 = ['PRP', 'VBG', 'VBZ', 'IN']
    l4 = ['PRP', 'VBG', 'VBZ']
    l5 = ['PRP', 'VBG', 'VBD']
    l6 = ['NNP', 'VBG', 'VBD']
    l7 = ['NN', 'VBG', 'VBZ']

    l8 = ['NNP', 'VBZ', 'JJ']
    l9 = ['NNP', 'VBZ', 'NN']

    l10 = ['NNP', 'VBZ']
    l11 = ['PRP', 'VBZ']
    l12 = ['NNP', 'NN', 'IN']
    l13 = ['NN', 'VBZ']
    
   
    # Question starting with WHAT
    if all(key in bucket for key in l1): #'NNP', 'VBG', 'VBZ', 'IN' in sentence.
        question = 'What' + ' ' + line.words[bucket['VBZ']] +' '+ line.words[bucket['NNP']]+ ' '+ line.words[bucket['VBG']] + '?'
        questions.append(question)

    
    elif all(key in  bucket for key in l2): #'NNP', 'VBG', 'VBZ' in sentence.
        question = 'What' + ' ' + line.words[bucket['VBZ']] +' '+ line.words[bucket['NNP']] +' '+ line.words[bucket['VBG']] + '?'
        questions.append(question)

    
    elif all(key in  bucket for key in l3): #'PRP', 'VBG', 'VBZ', 'IN' in sentence.
        question = 'What' + ' ' + line.words[bucket['VBZ']] +' '+ line.words[bucket['PRP']]+ ' '+ line.words[bucket['VBG']] + '?'
        questions.append(question)

    
    elif all(key in  bucket for key in l4): #'PRP', 'VBG', 'VBZ' in sentence.
        question = 'What ' + line.words[bucket['PRP']] +' '+  ' does ' + line.words[bucket['VBG']]+ ' '+  line.words[bucket['VBG']] + '?'
        questions.append(question)

    elif all(key in  bucket for key in l7): #'NN', 'VBG', 'VBZ' in sentence.
        question = 'What' + ' ' + line.words[bucket['VBZ']] +' '+ line.words[bucket['NN']] +' '+ line.words[bucket['VBG']] + '?'
        questions.append(question)

    elif all(key in bucket for key in l8): #'NNP', 'VBZ', 'JJ' in sentence.
        question = 'What' + ' ' + line.words[bucket['VBZ']] + ' ' + line.words[bucket['NNP']] + '?'
        questions.append(question)

    elif all(key in bucket for key in l9): #'NNP', 'VBZ', 'NN' in sentence
        question = 'What' + ' ' + line.words[bucket['VBZ']] + ' ' + line.words[bucket['NNP']] + '?'
        questions.append(question)

    elif all(key in bucket for key in l11): #'PRP', 'VBZ' in sentence.
        if line.words[bucket['PRP']] in ['she','he']:
            question = 'What' + ' does ' + line.words[bucket['PRP']].lower() + ' ' + line.words[bucket['VBZ']].singularize() + '?'
            questions.append(question)

    elif all(key in bucket for key in l10): #'NNP', 'VBZ' in sentence.
        question = 'What' + ' does ' + line.words[bucket['NNP']] + ' ' + line.words[bucket['VBZ']].singularize() + '?'
        questions.append(question)

    elif all(key in bucket for key in l13): #'NN', 'VBZ' in sentence.
        question = 'What' + ' ' + line.words[bucket['VBZ']] + ' ' + line.words[bucket['NN']] + '?'
        questions.append(question)

    # When the tags are generated 's is split to ' and s. To overcome this issue.
    if 'VBZ' in bucket and line.words[bucket['VBZ']] == "’":
        question = question.replace(" ’ ","'s ")
        questions.append(question)
        
    if "because" in sentence.lower() and line.words[bucket['VBZ']]:
        question = 'Why ' +  line.words[bucket['VBZ']]

        end_index = sentence.split().index("because")
        for i in range(end_index):
            if i!=bucket['VBZ']:
                question += (" " + line.words[i])

        question += (" " + "?")
        questions.append(question)
   
    return questions

# finding and sorting sentences with importance in corpus

def find_most_imp_sent():
    global tf_idf,psent
    
    most_imp_sent = {}
    max_score = 0
    for sent in psent:
        score = 0
        count = 0

        for word in sent:
            if word not in stopwords:
                count = count + 1
                score = score + tf_idf[word]
                
        if count==0:
            score = 0
        else:
            score /= count
        sentence = " ".join(sent)
        most_imp_sent[sentence] = score
    sorted_x = sorted(most_imp_sent.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = dict(sorted_x)
    return sorted_x
                    
def stanford_ner(filename,verbose=False, absolute_path=None):
    
    IS_WINDOWS = True if system() == 'Windows' else False
    JAVA_BIN_PATH = 'java.exe' if IS_WINDOWS else 'java'
    STANFORD_NER_FOLDER = 'stanford-ner'
    
    out = 'out.txt'
    command = ''
    if absolute_path is not None:
        command = 'cd {};'.format(absolute_path)
    else:
        filename = '../{}'.format(filename)

    command += 'cd {}; {} -mx1g -cp "*:lib/*" edu.stanford.nlp.ie.NERClassifierCombiner '                '-ner.model classifiers/english.all.3class.distsim.crf.ser.gz '                '-outputFormat tabbedEntities -textFile {} > ../{}'         .format(STANFORD_NER_FOLDER, JAVA_BIN_PATH, filename, out)

    if verbose:
        debug_print('Executing command = {}'.format(command), verbose)
        java_process = Popen(command, stdout=stderr, shell=True)
    else:
        java_process = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    java_process.wait()
    assert not java_process.returncode, 'ERROR: Call to stanford_ner exited with a non-zero code status.'

    if absolute_path is not None:
        out = absolute_path + out

    with open(out, 'r') as output_file:
        results_str = output_file.readlines()
    os.remove(out)

    results = []
    for res in results_str:
        if len(res.strip()) > 0:
            split_res = res.split('\t')
            entity_name = split_res[0]
            entity_type = split_res[1]

            if len(entity_name) > 0 and len(entity_type) > 0:
                results.append([entity_name.strip(), entity_type.strip()])

    if verbose:
        pickle.dump(results_str, open('out.pkl', 'wb'))
        debug_print('wrote to out.pkl', verbose)
    return results

#---------------------- MAIN -----------------------------------------

vocab = {}
words_sents_map = {}
total_words_in_sent = {}
tf_idf = {}

#psent = pre_process(data4)
psent = pre_process(data2)
#psent = pre_process(data1)

inverse_dict()
calc_tf_idf()

sentence = {}

sentence = find_most_imp_sent()

length_of_map = len(sentence)

length_of_map /= 2;

count = 0

for k,v in sentence.items():
    count += 1
    filename = "testfile.txt"
    file1 = open(filename,'w') 
    file1.write(k + ".")
    file1.close()
    
    ner = stanford_ner(filename)
    questions = genQuestion(k,ner)
    
    for question in questions:
        print("\nQuestion: " , question)
        print("Answer: ",k,".")       