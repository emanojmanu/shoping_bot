import json
import string
import random 
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer 
import tensorflow as tensorF
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
remove_stop_word = True

nltk.download("punkt")
nltk.download("wordnet")
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
    {"tag": "greeting",
    "patterns": [ "Hi", "Hello", "Hey"],
    "responses": ["Hello!!"],
    },
    {"tag": "name",
    "patterns": ["who are you?"],
    "responses": ["Much intelligent bot in parallel universe"]
    }
]}

def filter_stop_words(input_words):
    return [
        word for word in input_words if word not in stopwords
    ]

def setup_model():
    lm = WordNetLemmatizer() #for getting words
    # lists
    ourClasses = []
    trainWordCorpus = []
    documentX = []
    documentY = []
    # Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
    for intent in ourData["intents"]:
        for pattern in intent["patterns"]:
            ourTokens = nltk.word_tokenize(pattern)
            if remove_stop_word:
                ourTokens = filter_stop_words(ourTokens)
            trainWordCorpus.extend(ourTokens)
            documentX.append(pattern)
            documentY.append(intent["tag"])
        
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
        text = lm.lemmatize(doc.lower())
        for word in trainWordCorpus:
            bagOfwords.append(1) if word in text else bagOfwords.append(0)
        
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

