'''
Synonym handler
'''
import string
import random 
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer 
import tensorflow as tensorF
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from nltk.corpus import stopwords

from nltk.corpus import wordnet as wn

stopwords = stopwords.words('english')
remove_stop_word = True
relativeness_threshold = 2.26

nltk.download("punkt")
nltk.download("wordnet")
#nltk.download("stopword")
ourData = {"intents": [        
    {"tag": "enquiry",
    "patterns": ["How many pencils do we have?"],
    "responses": ["I need to check inventory"]
    },
    {"tag": "greeting",
    "patterns": [ "Hi", "Hello", "Hey"],
    "responses": ["Hi there", "Hello", "Hi :)"],
    },
    {"tag": "order",
    "patterns": ["I want to order 1000 pencils", "Place order for 500 shirts", "Purchase 99 badge holders"],
    "responses": ["I will place the order..."]
    },

]}


def get_synonyms(input_word):
  for net1 in wn.synsets(input_word):
    for net2 in wn.all_synsets():
        try:
            lch = net1.lch_similarity(net2)
        except:
            continue
        # The value to compare the LCH to was found empirically.
        if lch >= relativeness_threshold:
            yield net2.name().split(".")[0]

def get_total_sentence_possibilities(input_sentence):
  """Given an input sentence returns the total possible sentences."""
  all_possible_related_words = []
  print(input_sentence)
  for word in input_sentence:
    all_possible_related_words.append(word)
    for related_word in get_synonyms(word):
      all_possible_related_words.append(related_word)
  return all_possible_related_words


def filter_stop_words(input_words):
    return [
        word for word in input_words if word not in stopwords and word not in string.punctuation
    ]

def setup_model():
    lm = WordNetLemmatizer() #for getting words
    # lists
    ourClasses = []
    trainWordCorpus = set()
    documentX = []
    documentY = []
    document_train_words = []
    # Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
    for intent in ourData["intents"]:
        for pattern in intent["patterns"]:
            ourTokens = nltk.word_tokenize(pattern)
            if remove_stop_word:
                ourTokens = filter_stop_words(ourTokens)
            ourTokens = get_total_sentence_possibilities(ourTokens)

            document_train_words.append(ourTokens)
            documentX.append(pattern)
            documentY.append(intent["tag"])
            trainWordCorpus.update(ourTokens)

        if intent["tag"] not in ourClasses:# add unexisting tags to their respective classes
            ourClasses.append(intent["tag"])

    trainWordCorpus = [lm.lemmatize(word.lower()) for word in trainWordCorpus if word not in string.punctuation] # set words to lowercase if not in punctuation
    trainWordCorpus = sorted(set(trainWordCorpus))# sorting words
    ourClasses = sorted(set(ourClasses))# sorting classes
    print(trainWordCorpus)

    trainingData = [] # training list array
    outEmpty = [0] * len(ourClasses)
    # BOW model
    for idx, doc in enumerate(documentX):
        bagOfwords = []
        for word in trainWordCorpus:
            bagOfwords.append(1) if word in document_train_words[idx] else bagOfwords.append(0)  
        outputRow = list(outEmpty)
        outputRow[ourClasses.index(documentY[idx])] = 1
        trainingData.append([bagOfwords, outputRow])

    random.shuffle(trainingData)
    trainingData = num.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

    x = num.array(list(trainingData[:, 0]))# first trainig phase
    y = num.array(list(trainingData[:, 1]))# second training phase

    # defining some parameters
    iShape = (len(x[0]),)
    oShape = len(y[0])

    # the deep learning model
    ourNewModel = Sequential()
    ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
    ourNewModel.add(Dropout(0.5))
    ourNewModel.add(Dense(64, activation="relu"))
    ourNewModel.add(Dropout(0.3))
    ourNewModel.add(Dense(oShape, activation = "softmax"))
    md = tensorF.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    ourNewModel.compile(loss='categorical_crossentropy',
                optimizer=md,
                metrics=["accuracy"])

    ourNewModel.fit(x, y, epochs=200, verbose=1)
    return (trainWordCorpus, ourClasses, ourNewModel)

def ourText(text): 
  newtkns = nltk.word_tokenize(text)
  lm = WordNetLemmatizer()
  newtkns = [lm.lemmatize(word) for word in newtkns]
  if remove_stop_word:
    newtkns = [
        tkn for tkn in newtkns if tkn not in stopwords
    ]
  #newtkns = get_total_sentence_possibilities(newtkns)
  return newtkns

def wordBag(text, vocab): 
  newtkns = ourText(text)
  bagOwords = [0] * len(vocab)
  for w in newtkns: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bagOwords[idx] = 1
  return num.array(bagOwords)

def pred_class(text, vocab, labels, inputModel): 
  bagOwords = wordBag(text, vocab)
  ourResult = inputModel.predict(num.array([bagOwords]))[0]
  newThresh = 0.2
  yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

  yp.sort(key=lambda x: x[1], reverse=True)
  newList = []
  for r in yp:
    newList.append(labels[r[0]])
  return newList

def getRes(firstlist, fJson): 
  tag = firstlist[0]
  listOfIntents = fJson["intents"]
  for i in listOfIntents: 
    if i["tag"] == tag:
      ourResult = random.choice(i["responses"])
      break
  return ourResult

def start_listening(newMessage, words, outputClasses, inputModel):
    intents = pred_class(newMessage, words, outputClasses, inputModel)
    ourResult = getRes(intents, ourData)
    return ourResult

