#Joshua Castro and Trevor Hitchcock
# November-December 2023
# Dr. Silveyra CS
# This project satisfies the requirements for the final project, with analysis using the Cranfield Documents
#imports/downloads
import nltk
import re
import requests
import time
import json
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#global variable to stop program at desired number of pages visited, as well as maximum (was 10,000, but only 4955 on Muhlenberg, so hard coded to not mess up files)
numVis = 0
numMax = 4955


#method to crawl all pages, starting from startUrl
#param: string - url to start crawling from
#return: none
def crawl(startUrl):
  # checks if already crawled from startUrl

  with open("page1.txt", 'r', encoding='utf-8') as file:
    url = next(file).split()[1]
    if url == startUrl:
      print("Already crawled starting at this page")
      return

  # reads robots.txt file
  rp = RobotFileParser()
  rp.set_url(urljoin(startUrl, '/robots.txt'))
  rp.read()
  # grabs crawl delay
  crawlDelay = rp.crawl_delay("*") or 1  # 1 if not specified
  crawlPage(startUrl, [startUrl], crawlDelay, rp)
  print("Done crawling")


# method to crawl a page recursively, calls itself if an unseen link that is an html page is found in the current page
# param: url - string containing url to crawl, seen - list of websites already visited, cd - int for delay, rp - RobotFileParser object to check robots.txt
# return: none
def crawlPage(url, seen, cd, rp):
  # checks if the URL can be fetched according to robots.txt rules
  if not rp.can_fetch("*", url):
    return  # if not allowed, don't crawl
  # visit page, and then delay
  global numVis
  numVis += 1
  time.sleep(cd)
  # gets text of page
  text = requests.get(url).text
  # Extract links using regular expression
  rL, aL = getLinks(text)
  # Write to file
  writeToFile(url, text, rL, aL)
  # Recursively crawl relative links
  for link in rL:
    relUrl = urljoin(url, link)
    # checks to ensure link is html and not already visited

    if not relUrl.endswith(".svg") and not relUrl.endswith(
        ".png"
    ) and not relUrl.endswith(".webmanifest") and not relUrl.endswith(
        ".css"
    ) and "icons.svg" not in relUrl and not relUrl.endswith(
        ".ico"
    ) and not relUrl.endswith(
        ".pdf"
    ) and "email-protection#" not in relUrl and not relUrl.endswith(
        ".php"
    ) and "/media" not in relUrl and "www.li" not in relUrl and "dev.fastspot.com" not in relUrl and relUrl not in seen and numVis < numMax:
      seen.append(relUrl)
      # add link to seen and call crawl method on it
      crawlPage(relUrl, seen, cd, rp)


# method to clean the contents of a file
# param: content - string containing all html content of page, remSW / lem - boolean flags to determine if lemming and removing stopwords, link - boolean flag if the page has a link (false if query)
# return: string with cleaned text
def clean(content, remSW, lem, link):
  # if link (not query)
  if link:
    x = re.findall(">([^<]+)</[^>]+>", content)
    x = ' '.join([a.lower() for a in x if a != ''])
  # if query, sets to lowercase
  else:
    x = content.lower()
  # Parse the HTML content
  # remove characters, markup chars (amp, nbsp, etc), and more
  charsub = re.sub(
      r"[\'\\\!\?\=\+\_\%\|\\\/\(\)\@\#\`\<\>\~\&\^\;\:\,\[\]\{\}\*\â\$]", r"",
      x)
  spacesub = re.sub(r"[\[\"\.]", " ", charsub)
  htmlsub = re.sub(r"\bamp\b|nbsp|rsquo|mdash", r" ", spacesub)
  dashsub = re.sub(r"[-—]", r" ", htmlsub)
  # replace numbers with <number>
  s = re.sub(r"\d+", r"<number>", dashsub)
  # removes stopwords and lemms text based on flags, returning cleaned text in each case
  if remSW and lem:
    SW = stopwords.words("english")
    wnl = WordNetLemmatizer()
    ret = ""
    for w in s.split():
      w = wnl.lemmatize(w)
      if w not in SW:
        ret += (w + " ")
    return ret
  elif remSW:
    ret = ""
    SW = stopwords.words("english")
    for w in s.split():
      if w not in SW:
        ret += (w + " ")
    return ret
  elif lem:
    wnl = WordNetLemmatizer()
    ret = ""
    for w in s.split():
      ret += wnl.lemmatize(w) + " "
    return ret
  return s


# method to get the links from uncleaned html content of a website
# param: text - string containing uncleaned html content
# return: tuple containing a list of relative links and a list of absolute links
def getLinks(text):
  # Extract links using regular expression
  links = re.findall(r'href\=[\"\'](https?\:\/\/[^\s\'"]+|/[^\s\'"]+)', text)
  # stores relative and absolute links appropriately
  relativeLinks = []
  absoluteLinks = []
  for link in links:
    if link.startswith('http'):
      absoluteLinks.append(link)
    else:
      relativeLinks.append(link)
  return (relativeLinks, absoluteLinks)


# method to write html text to a file
# param: url - string containing url of the page, text - string containing the html text of a file
# return: none
def writeToFile(url, text):
  filename = "page" + str(numVis) + ".txt"
  with open(filename, 'w', encoding='utf-8') as file:
    file.write("URL: " + url + "\n")
    file.write(text)
  print("Page " + url + " crawled and saved to " + filename)


# method to create an inverted index
# param: remSW/lem - boolean flags to determine if user wants to remove SW and/or lem content
# return: inverted index, max frequency array
def invertedIndex(remSW, lem):
  # blank dictionary and list for inverted index and max frequency
  index = {}
  maxFreq = []
  # loops through each .txt file containing the raw html contents of each page
  for i in range(1, numMax + 1):
    # opens file
    filename = "page" + str(i) + ".txt"
    with open(filename, 'r', encoding='utf-8') as file:
      # reads in each word, after skipping the line containing the url, cleans, and calls index file variable.
      link = next(file).split()[1].strip()
      c = clean(file.read(), remSW, lem, True)
      index, m = indexFile(i, c, index)
      # updates max frequency array, closes file
      maxFreq.append((link, m))
    file.close()
  return (index, maxFreq)


#method to update inverted index for a single file
# param: fileNum - int number of the file, cleanContents - string containing clean contents of the file, index - copy of inverted index dictionary
# return: index - updated inverted index dictionary, m - maximum word count in file, for maxFreq array
def indexFile(fileNum, cleanContents, index):
  m = 1
  # goes word by word in the clean contents
  for word in cleanContents.split():
    # if we have not seen the word before, creates a dictionary for the word
    if word not in index:
      index[word] = {fileNum: 1}
    # if we have seen the word before in this file, updates count in dictionary, updates max if applicable
    elif fileNum in index[word]:
      index[word][fileNum] += 1
      m = max(m, index[word][fileNum])
    # if we have seen the word before but not in this file, adds a new entry in the dictionary within the word dictionary for the file number
    else:
      index[word][fileNum] = 1
  return (index, m)


# method to calculate the tfidf of a word
# param: word - string containing a word, index - dictionary of inverted index, maxF - list of max frequency in each document
# return: tfidfList - list containing the tfidf of the word in each document
def calcTfidf(word, index, maxF):
  #calculate the TFIDF of the word for each document, including query
  tfidfList = []
  idf = 1 + math.log((numMax + 1) / len(index[word]), 10)
  # loops through each document and query, adding tfidf if the word exists in the document, if not, adds 0
  for docNum in range(1, numMax + 2):
    if docNum in index[word]:
      tfidfList.append((index[word][docNum] / maxF[docNum - 1][1]) * idf)
    else:
      tfidfList.append(0)
  return tfidfList


# method to calculate the cosine similarity for every document compared to the query
# param: tfidf - dictionary with list of tfidfs for each word and document
# return: cosSim - list of cosSim score for each document
def cosSim(tfidf):
  cosSim = {}
  # for each document
  for i in range(0, numMax):
    num = 1
    aSum = 1
    bSum = 1
    # for each word in the query, calculate tfidf compared to the query
    for word in tfidf:
      a = tfidf[word][i]
      b = tfidf[word][numMax]
      num += (a * b)
      aSum += (a * a)
      bSum += (b * b)
    den = math.sqrt(aSum) * math.sqrt(bSum)
    cosSim[i] = (num / den)
  return cosSim


# method to display the top 10 document links
# param: sCS - sorted list of cosine similaries, linkArr - array which contains the link of each document
# return: none
def display(sCS, linkArr):
  # loops 10 times
  for i in range(10):
    # linkarr[sCS[i][0]][0] contains the document link.
    print((i + 1), ".", linkArr[sCS[i][0]][0])


# main method to run program
def main():
  # https://www1.lehigh.edu/
  # https://www.muhlenberg.edu/
  startUrl = "https://www.muhlenberg.edu/"
  crawl(startUrl)
  lem = input("Do you want to lemmatize the words? (y/n):")
  sw = input("Do you want to remove stopwords? (y/n):")
  # create inverted index
  print("Indexing...")
  index, freqArr = invertedIndex(lem == "y", sw == "y")
  # add query as a new document
  query = input("Indexing Completed. Please enter a query: ")
  q = clean(query, lem == "y", sw == "y", False)
  index, m = indexFile(numMax + 1, q, index)
  freqArr.append(("query", m))
  tfidf = {}
  for word in q.split():
    if word not in tfidf:
      tfidf[word] = calcTfidf(word, index, freqArr)
  #calculate cosine similarity for every document except query
  cs = cosSim(tfidf)
  # sorts keys using lambda function
  sortedCosSimKeys = sorted(cs.items(), key=lambda x: x[1], reverse=True)
  # sort the documents by cosine similarity, putting the keys in a sorted list by value
  display(sortedCosSimKeys, freqArr)
  # loops while user wants to query more
  query = input("Enter another query, or q to quit: ")
  while query != "q":
    q = clean(query, lem == "y", sw == "y", False)
    index, m = indexFile(numMax + 1, q, index)
    freqArr[-1] = ("query", m)
    # dictionary containing "word" keys with a list of tfidfs for the word in each document as value
    tfidf = {}
    for word in q.split():
      if word not in tfidf:
        tfidf[word] = calcTfidf(word, index, freqArr)
    #calculate cosine similarity for every document except query
    cs = cosSim(tfidf)
    sortedCosSimKeys = sorted(cs.items(), key=lambda x: x[1], reverse=True)
    # sort the documents by cosine similarity, putting the keys in a sorted list by value
    display(sortedCosSimKeys, freqArr)
    query = input("Enter another query, or q to quit: ")
  print("goodbye.")


main()
