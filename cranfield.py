#Joshua Castro and Trevor Hitchcock
# November-December 2023
# Dr. Silveyra CS
# This project satisfies the requirements for Step 5 of the final project, with an analysis report using the Cranfield Documents
#imports/downloads
import re
import math
import copy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



#method to open the cran.all.1400 file and read in the contents, storing them as appropriate
#param: none
#return: documents - dictionary containing a dictionary key for each document that contains the title and text of the document
def crawl():
  print("running")
  documents = {}
  with open("cran.all.1400", 'r', errors='ignore') as file:
    contents = file.read()
    file.close()
  # split the contents into chunks every time a new document appears (.I), starting at position 1 because position 0 contains text until the first ".I"
  docs = contents.split(".I")[1:]
  # for each document
  for doc in docs:
    # replace line breaks with spaces, and then re-add line breaks between each section (.T, .A, .B, .W), setting this into a list in the form [doc number,title,author,bibliography,text)
    info = doc.replace("\n", " ").replace(".T", "\n").replace(".A","\n").replace(".B","\n").replace(".W","\n").split("\n")
    # gets relevant info from list (docNum, title, text), also strips to get rid of added in line breaks and any leading or trailing blank spaces
    docNum = int(info[0].strip().replace(" ",""))
    title = info[1].strip().replace(".","")
    contents = info[4].strip()
    #updates dictionary
    documents[docNum] = {"Title": title, "Text": contents}
  return documents


# method to clean the contents of a string
# param: content - string containing the raw text of a document, remSW / lem - boolean flags to determine if lemming and removing stopwords
# return: string with cleaned text
def clean(content, remSW, lem):
  x = content.lower()
  # remove characters
  charsub = re.sub(r"[\'\\\!\?\=\+\_\%\|\\\/\(\)\@\#\`\<\>\~\&\^\;\:\,\[\]\{\}\*\â\$\[\"\.\-\—]", r"", x)
  # replace numbers with <number>
  s = re.sub(r"\d+", r"<number>", charsub)
  # removes stopwords and lemms text based on flags, returning final cleaned text in each case 
  if remSW and lem:
    SW = stopwords.words("english")
    wnl = WordNetLemmatizer()
    ret = ""
    for w in s.split():
      w = wnl.lemmatize(w)
      if w not in SW:
        ret += (w + " ")
    return ret.strip()
  elif remSW:
    ret = ""
    SW = stopwords.words("english")
    for w in s.split():
      if w not in SW:
        ret += (w + " ")
    return ret.strip()
  elif lem:
    wnl = WordNetLemmatizer()
    ret = ""
    for w in s.split():
      ret += wnl.lemmatize(w) + " "
    return ret.strip()
  return s.strip()

# method to create an inverted index from a given dictionary of documents
# param: dict - dictionary containing an entry for each document
# return: index - inverted index of words, maxFreq - list containing the maximum frequency of each document
def invertedIndex(dict):
  # blank dictionary and list for inverted index and max frequency
  index = {}
  maxFreq = []
  # loops through each document in the dictionary 
  for doc in dict:
    m = 1
    # for each word in the document text
    for word in dict[doc]["Text"].split():
      # if we have not seen the word before, creates a dictionary for the word within the larger dictionary
      if word not in index:
        index[word] = {doc: 1}
      # if we have seen the word before, and have seen it in this document, updates count in dictionary and updates max if applicable
      elif doc in index[word]:
        index[word][doc] += 1
        m = max(m, index[word][doc])
      # if we have seen the word before but not in this document, adds a new entry in the dictionary within the word dictionary for the document number
      else:
        index[word][doc] = 1  
    # adds maximum frequency of the document at end of max frequency list, also storing the document number and title of the document
    maxFreq.append(((doc-1,dict[doc]["Title"]), m)) 
  return (index, maxFreq)

# method to update our inverted index every time a new query is entered/read-in
# param: index - dictionary containing our original inverted index, freqArr - list containing our original frequency list, query - string containing the cleaned text of the query
# return: index - inverted index of words, maxFreq - list containing the maximum frequency of each document, both with the query added in docNum 1401
def updateIndex(index, freqArr, query):
  m = 1
  # same process as for each file, reads in each word of query, storing it in docNum = 1401 in the index
  for word in query.split():
      # if we have not seen the word before, creates a dictionary for the word within the larger dictionary
      if word not in index:
        index[word] = {1401: 1}
      # if we have seen the word before, and have seen it in the query, updates count in dictionary and updates max if applicable
      elif 1401 in index[word]:
        index[word][1401] += 1
        m = max(m, index[word][1401])
      # if we have seen the word before but not in the query, adds a new entry in the dictionary within the word dictionary for the document number
      else:
        index[word][1401] = 1 
  # updates last entry of max frequency array, which corresponds to the query, including the "document number" that we set to be 1401 as well as the text of the query
  freqArr[-1] = ((1401,"query"), m)
  return (index, freqArr) 

# method to get every query out of the cran.qry file
# param: remSW / lem - boolean flags to determine if lemming and removing stopwords
# return: queries - list containing the cleaned text of each query
def getQs(lem, SW):
  # creates blank list, then reads in contents of the file
  queries = []
  with open("cran.qry", 'r', errors='ignore') as file:
    contents = file.read()
    file.close()
  # same as with cran.all.1400, splits into a list containing each query
  docs = contents.split(".I")[1:]
  querNum = 0
  # for each query, updates number of queries stored, removes linebreaks, adds a linebreak between the number and text contents, and stores the cleaned contents of the text with the query number
  for doc in docs:
    querNum+=1
    info = doc.replace("\n", " ").replace(".W","\n").split("\n")
    queries.append((clean(info[1].strip(), lem, SW),querNum))
  return queries

# method to calculate the tfidf of a word
# param: word - string containing a word, index - dictionary of inverted index, maxF - list of max word frequency in each document
# return: tfidfList - list containing the tfidf of the word in each document
def calcTfidf(word, index, maxF):
  #calculate the TFIDF of the word for each document, including the query
  tfidfList = [0]
  idf = 1 + math.log(len(maxF) / len(index[word]), 10)
  # loops through each document and query, adding tfidf if the word exists in the document, if not, adds 0 
  for docNum in range(1, len(maxF)+1):
    if docNum in index[word]:
      tfidfList.append((index[word][docNum] / maxF[docNum - 1][1]) * idf)
    else:
      tfidfList.append(0)
  return tfidfList

# method to calculate the cosine similarity for every document compared to the query
# param: tfidf - dictionary with list of tfidfs for each word and document, including query
# return: cosSim - list of cosSim score for each document
def cosSim(tfidf):
  cosSim = {}
  # for each document, excluding query, calculates each word tfidf compared to query
  for i in range(1, 1401):
    num = 1
    aSum = 1
    bSum = 1
    # for each word in the query, calculate tfidf compared to the query, add it to the cosSim list
    for word in tfidf:
      a = tfidf[word][i]
      b = tfidf[word][1401]
      num += (a * b)
      aSum += (a * a)
      bSum += (b * b)
    den = math.sqrt(aSum) * math.sqrt(bSum)
    cosSim[i] = (num / den)
  return cosSim

# method to extract the relevant documents for each query from the cranqrel file
# param: queries - list containing the cleaned text of each query 
# return: relevant - dictionary containing the relevant documents for each query
def getRelevantDocs(queries):
  relevant = {}
  # opens file and reads in content
  with open("cranqrel", 'r', errors='ignore') as file:
    contents = file.read()
    file.close()
  # creates a dictionary entry for each query 
  for q in queries:
    relevant[q[1]] = {"pos": [], "top": 0}
  # for each line, gets the query, relevant document, and weight
  for line in contents.split("\n"):
    lineList = line.split()
    qNum = int(lineList[0])
    dNum = int(lineList[1])
    weight = int(lineList[2])
    # if weight is greater than 0, adds it to positive connection list, if weight is -1, adds it to the top key as it is the most relevant document
    if weight > 0:
      relevant[qNum]["pos"].append(dNum)
    else:
      relevant[qNum]["top"] = dNum
  return relevant



# main method to run program
def main():
  # get the title and contents of each document in a dictionary
  contentDict = crawl()
  # receives user input to determine if they want to remove stopwords or lem
  lem = input("Do you want to lemmatize the words? (y/n):")
  sw = input("Do you want to remove stopwords? (y/n):")
  print("Indexing...")
  # cleans the text of each document, overwriting entry in document dictionary,  creates the inverted index from the cleaned text
  for doc in contentDict:
    contentDict[doc]["Text"] = clean(contentDict[doc]["Text"],lem == "y", sw == "y")
  # create inverted index
  index, freqArr = invertedIndex(contentDict)
  # gets clean text for each query and relevant documents for each query
  print("Indexing Completed. Now Querying")
  queries = getQs(lem == "y", sw == "y")
  relevant = getRelevantDocs(queries)
  freqArr.append(((1401,"query"), 1))
  # variables to track performance
  totTrueP = 0
  totFalseP = 0
  totTrueN = 0
  totFalseN = 0
  bestFit = 0
  bestBest = 0
  # for each query
  for query in queries:
    # creates a copy of our inverted index (because we don't want to include the word count of words in other queries)
    indexCop = copy.deepcopy(index)
    trueP = 0
    falseP = 0
    # adds query to index and frequency list
    tempIndex, freqArr = updateIndex(indexCop, freqArr, query[0])
    # calculates tfidf of each word in query
    tfidf = {}
    for word in query[0].split():
      if word not in tfidf:
        tfidf[word] = calcTfidf(word, tempIndex, freqArr)
    #calculate cosine similarity for every document except query, sorting the results
    cs = cosSim(tfidf)
    sortedCosSimKeys = sorted(cs.items(), key = lambda x: x[1], reverse=True)
    # gets the relevant documents for the query
    possibleMatches = relevant[query[1]]["pos"] + [relevant[query[1]]["top"]]
    # checks the top i matches, with i being the count of relevant documents for the query, updating counter variables
    for i in range (len(possibleMatches)):
      docNum = freqArr[sortedCosSimKeys[i][0]][0][0]
      if docNum == relevant[query[1]]["top"]:
        if i == 0:
          bestBest += 1
        bestFit += 1
        trueP += 1
      elif docNum in relevant[query[1]]["pos"]:
        trueP += 1
      else:
        falseP += 1
    # updates overall counter variables
    totTrueP += trueP
    totFalseP += falseP
    totTrueN += (1400 - len(possibleMatches) - falseP)
    totFalseN += (len(possibleMatches) - trueP)
  # after reading in each query, prints results
  print("True Positives:",totTrueP)
  print("False Positives:",totFalseP)
  print("True Negatives:",totTrueN)
  print("False Negatives:",totFalseN)
  print("Best Matches Found:",bestFit)
  print("Best Matches Found in Top Spot:",bestBest)


main()
