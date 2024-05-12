---
title: "Text Analysis"
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

- What are text analysis methods?
- How can I perform text analysis?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Define objectives associated with each one of the text analysis techniques.
- Implement named entity recognition, and topic modeling using Python libraries and frameworks, such as NLTK, and Gensim.

::::::::::::::::::::::::::::::::::::::::::::::::



## 3.1. Introduction to Text-Analysis

In this episode, we will learn how to analyze text data for NLP tasks. We will explore some common techniques and methods for text analysis, such as named entity recognition, topic modeling, and text summarization. We will use some popular libraries and frameworks, such as spaCy, NLTK, and Gensim, to implement these techniques and methods.


::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: What are some of the goals of text analysis for NLP tasks in your research field (e.g. material science)? Think of some examples of NLP tasks that require text analysis, such as literature review, patent analysis, or material discovery. How does text analysis help to achieve these goals?

:::::::::::::::::::::::::::::::::::::::::::::::




::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: Name some of the common techniques in text analysis and associating libraries. Briefly explain how they differ from each other in terms of their objectives and required libraries.

:::::::::::::::::::::::::::::::::::::::::::::::



## 3.1. Named Entity Recognition

Named Entity Recognition is a process of identifying and classifying key elements in text into predefined categories. The categories could be names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. Next, let’s discuss how it works. 


::::::::::::::::::::::::::::::::::::: challenge

## Discussion

Teamwork: Discuss what tasks can be done with NER.

A: *NER can help with 1) categorizing resumes, 2) categorizing customer feedback, 3) categorizing research papers, etc.*

:::::::::::::::::::::::::::::::::::::::::::::::


Using a text example from Wikipedia can help us to see how NER works. Note that the spaCy library is a common framework here as well. Thus, first, we make sure that the library is installed and imported:

```python

pip install spacy
import spacy


```

Create an NLP model (*nlp*) and download the small English model from spaCy that is suitable for general tasks.



```python

nlp = spacy.download("en_core_web_sm")

```

Create a variable to store your text and then apply the model to process your text (text from [Wikipedia](https://en.wikipedia.org/wiki/Australian_Securities_Exchange)):

:::::::::::::::::::::::::::::::::::::::::: spoiler

### Text

text = “Australian Shares Exchange Ltd (ASX) is an Australian public company that operates Australia's primary shares exchange, the Australian Shares Exchange (sometimes referred to outside of Australia as, or confused within Australia as, The Sydney Stock Exchange, a separate entity). The ASX was formed on 1 April 1987, through incorporation under legislation of the Australian Parliament as an amalgamation of the six state securities exchanges, and merged with the Sydney Futures Exchange in 2006. Today, ASX has an average daily turnover of A$4.685 billion and a market capitalization of around A$1.6 trillion, making it one of the world's top 20 listed exchange groups, and the largest in the southern hemisphere. ASX Clear is the clearing house for all shares, structured products, warrants and, ASX Equity Derivatives.”

::::::::::::::::::::::::::::::::::::::::::::::::::


Use for loop to print all the named entities in the document:


```python

doc = nlp(text)

For ent in doc.ents
	Print(ent.text, ent.label_)

```

The results will be:

```

output:

Australian Shares Exchange Ltd ORG
ASX ORG
Australian NORP
Australia GPE
the Australian Shares Exchange ORG
Australia GPE
Australia GPE
The Sydney Stock Exchange ORG
ASX ORG
1 April 1987 DATE
the Australian Parliament ORG
six CARDINAL
the Sydney Futures Exchange ORG
2006 DATE
Today DATE
ASX ORG
A$4.685 billion MONEY
around A$1.6 trillion MONEY
20 CARDINAL

```



::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: How can you interpret the labels in the output?

:::::::::::::::: solution

A: You can use the following code to get information about each one of the labels. For example, from we want to know what GPE represents here. We can use *explain()* to get the required information:
spacy.explain(‘GPE’)

```python

spacy.explain(‘GPE’)

```

```
Output: ‘Countries, cities, states’

```

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::





::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: Can we also use other libraries for NER analysis? Use the NLTK library to perform named entity recognition on the following text:

:::::::::::::::::::::::::::::::::::::::::: spoiler

### Text

text = " Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers. Additionally, perovskite nanocrystals exhibit high photoluminescence efficiencies, meaning they can efficiently convert absorbed light into emitted light, further adding to their potential for various optoelectronic applications.” 

::::::::::::::::::::::::::::::::::::::::::::::::::


Print the original text and the list of named entities and their types. You can use the following code to load the NLTK library and the pre-trained model for named entity recognition:

```python

import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
...

```


:::::::::::::::: solution

A: 
Download necessary NLTK resources and import the required toolkit:


```python

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

```

Store the text:

:::::::::::::::::::::::::::::::::::::::::: spoiler
### Text

text = "Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers. Additionally, perovskite nanocrystals exhibit high photoluminescence efficiencies, meaning they can efficiently convert absorbed light into emitted light, further adding to their potential for various optoelectronic applications."

::::::::::::::::::::::::::::::::::::::::::::::::::

```python

# Tokenize the text:
tokens = nltk.word_tokenize(text)

# Assign part-of-speech tags:
pos_tags = nltk.pos_tag(tokens)

# Perform named entity recognition:
named_entities = nltk.ne_chunk(pos_tags)

# Print the original text:
print("Original Text:")
print(text)

# Print named entities and their types:
print("\nNamed Entities:")
for entity in named_entities:
   if type(entity) == nltk.Tree:
   print(f"Entity: {''.join(word for word, _ in entity.leaves())}, Type: {entity.label()}")

```


:::::::::::::::::::::::::::::::::::::::::: spoiler
### Output

Original Text:
Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers. Additionally, perovskite nanocrystals exhibit high photoluminescence efficiencies, meaning they can efficiently convert absorbed light into emitted light, further adding to their potential for various optoelectronic applications.

- Named Entities:
- Entity: Perovskite, Type: ORGANIZATION
- Entity: light-emitting diodes (LEDs), Type: ORGANIZATION

[(('Perovskite', 'NNP'), 'ORGANIZATION'), ('nanocrystals', 'NNP'), ('have', 'VBP'), ('emerged', 'VBD'), ('as', 'IN'), ('a', 'DT'), ('promising', 'JJ'), ('class', 'NN'), ('of', 'IN'), ('materials', 'NNS'), ('for', 'IN'), ('next-generation', 'JJ'), ('optoelectronic', 'JJ'), ('devices', 'NNS'), ('due', 'IN'), ('to', 'TO'), ('their', 'PRP$'), ('unique', 'JJ'), ('properties', 'NNS'), ('.', '.'), ('Their', 'PRP$'), ('crystal', 'NN'), ('structure', 'NN'), ('allows', 'VBZ'), ('for', 'IN'), ('tunable', 'JJ'), ('bandgaps', 'NNS'), (',', ','), ('which', 'WDT'), ('are', 'VBP'), ('the', 'DT'), ('energy', 'NN'), ('differences', 'NNS'), ('between', 'IN'), ('occupied', 'VBN'), ('and', 'CC'), ('unoccupied', 'VBN'), ('electronic', 'JJ'), ('states', 'NNS'), ('.', '.'), ('This', 'DT'), ('tunability', 'NN'), ('enables', 'VBZ'), ('the', 'DT'), ('creation', 'NN'), ('of', 'IN'), ('materials', 'NNS'), ('that', 'WDT'), ('can', 'MD'), ('absorb', 'VB'), ('and', 'CC'), ('emit', 'VB'), ('light', 'NN'), ('across', 'IN'), ('a', 'DT'), ('wide', 'JJ'), ('range', 'NN'), ('of', 'IN'), ('the', 'DT'), ('electromagnetic', 'JJ'), ('spectrum', 'NN'), (',', ','), ('making', 'VBG'), ('them', 'PRP'), ('suitable', 'JJ'), ('for', 'IN'), ('applications', 'NNS'), ('like', 'IN'), ('solar', 'JJ'), ('cells', 'NNS'), (',', ','), ('light-emitting', 'JJ'), ('diodes', 'NNS'), ('(', '(', 'LEDs', 'NNPS'), ')', ')'), ('and', 'CC'), ('lasers', 'NNS'), ('.', '.'), ('Additionally', 'RB'), (',', ','), ('perovskite', 'NNP'), ('nanocrystals', 'NNP'), ('exhibit', 'VBP'), ('high', 'JJ'), ('photoluminescence', 'NN'), ('efficiencies', 'NNS'), (',', ','), ('meaning', 'VBG'), ('they', 'PRP'), ('can', 'MD'), ('efficiently', 'RB'), ('convert', 'VB'), ('absorbed', 'VBN'), ('light', 'NN'), ('into', 'IN'), ('emitted', 'VBN'), ('light', 'NN'), (',', ','), ('further', 'RB'), ('adding', 'VBG'), ('to', 'TO'), ('their', 'PRP$'), ('potential', 'NN'), ('for', 'IN'), ('various', 'JJ'), ('optoelectronic', 'JJ'), ('applications', 'NNS'), ('.', '.')]

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



::: callout

### Why NER?

When do we need to perform NER for your research? 


NER helps in quickly finding specific information in large datasets, which is particularly useful in research fields for categorizing the text based on the entities. NER is also called entity chunking and entity extraction.

:::


## 3.2. Topic Modeling

Topic Modeling is an unsupervised model for discovering the abstract “topics” that occur in a collection of documents. It is useful in understanding the main themes of a large corpus of text. To better understand this and to find the connection between concepts we have learned so far, let’s match the following terms to their brief definitions:



::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: To better understand this and to find the connection between concepts we have learned so far, let’s match the following terms to their brief definitions:

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/2be22cfb-4ed8-4f65-85bf-31962a40835f)


:::::::::::::::::::::::::::::::::::::::::::::::


::: callout

There are some new concepts in this section that are new to you. Although a detailed explanation of these concepts is out of the scope of this workshop we learn their basic definitions. You already learned a few of them in the earlier activity. Another one is Bag-of-words. We will learn more about Bag-of-words (BoW) in episode 5. BoW is defined as a representation of text that describes the occurrence of words within a document. It is needed in the topic modeling analysis to view the frequency of the words in a document regardless of the order of the words in the text.

:::


To see how Topic Modeling can help us in action to classify a text, let’s see the following example. We need to install the *Gensim* library and import the necessary modules:


```python

pip install genism
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess


```

at this stage, we preprocess including the text tokenization (text from [Wikipedia](https://en.wikipedia.org/wiki/Australian_Securities_Exchange)):

:::::::::::::::::::::::::::::::::::::::::: spoiler
### Text

text = "Australian Shares Exchange Ltd (ASX) is an Australian public company that operates Australia's primary shares exchange, the Australian Shares Exchange (sometimes referred to outside of Australia as, or confused within Australia as, The Sydney Stock Exchange, a separate entity). The ASX was formed on 1 April 1987, through incorporation under legislation of the Australian Parliament as an amalgamation of the six state securities exchanges and merged with the Sydney Futures Exchange in 2006. Today, ASX has an average daily turnover of A$4.685 billion and a market capitalisation of around A$1.6 trillion, making it one of the world's top 20 listed exchange groups, and the largest in the southern hemisphere. ASX Clear is the clearing house for all shares, structured products, warrants and ASX Equity Derivatives.
ASX Group[3] is a market operator, clearing house and payments system facilitator. It also oversees compliance with its operating rules, promotes standards of corporate governance among Australia's listed companies and helps to educate retail investors. Australia's capital markets. Financial development – Australia was ranked 5th out of 57 of the world's leading financial systems and capital markets by the World Economic Forum; Equity market – the 8th largest in the world (based on free-float market capitalisation) and the 2nd largest in Asia-Pacific, with A$1.2 trillion market capitalisation and average daily secondary trading of over A$5 billion a day; Bond market – 3rd largest debt market in the Asia Pacific; Derivatives market – largest fixed income derivatives in the Asia-Pacific region; Foreign exchange market – the Australian foreign exchange market is the 7th largest in the world in terms of global turnover, while the Australian dollar is the 5th most traded currency and the AUD/USD the 4th most traded currency pair; Funds management – Due in large part to its compulsory superannuation system, Australia has the largest pool of funds under management in the Asia-Pacific region, and the 4th largest in the world. Its primary markets are the AQUA Markets. Regulation. The Australian Securities & Investments Commission (ASIC) has responsibility for the supervision of real-time trading on Australia's domestic licensed financial markets and the supervision of the conduct by participants (including the relationship between participants and their clients) on those markets. ASIC also supervises ASX's own compliance as a public company with ASX Listing Rules. ASX Compliance is an ASX subsidiary company that is responsible for monitoring and enforcing ASX-listed companies' compliance with the ASX operating rules. The Reserve Bank of Australia (RBA) has oversight of the ASX's clearing and settlement facilities for financial system stability. In November 1903 the first interstate conference was held to coincide with the Melbourne Cup. The exchanges then met on an informal basis until 1937 when the Australian Associated Stock Exchanges (AASE) was established, with representatives from each exchange. Over time the AASE established uniform listing rules, broker rules, and commission rates. Trading was conducted by a call system, where an exchange employee called the names of each company and brokers bid or offered on each. In the 1960s this changed to a post system. Exchange employees called "chalkies" wrote bids and offers in chalk on blackboards continuously, and recorded transactions made. The ASX (Australian Stock Exchange Limited) was formed in 1987 by legislation of the Australian Parliament which enabled the amalgamation of six independent stock exchanges that formerly operated in the state capital cities. After demutualisation, the ASX was the first exchange in the world to have its shares quoted on its own market. The ASX was listed on 14 October 1998.[7] On 7 July 2006 the Australian Stock Exchange merged with SFE Corporation, holding company for the Sydney Futures Exchange. Trading system. ASX Group has two trading platforms – ASX Trade,[12] which facilitates the trading of ASX equity securities and ASX Trade24 for derivative securities trading. All ASX equity securities are traded on screen on ASX Trade. ASX Trade is a NASDAQ OMX ultra-low latency trading platform based on NASDAQ OMX's Genium INET system, which is used by many exchanges around the world. It is one of the fastest and most functional multi-asset trading platforms in the world, delivering latency down to ~250 microseconds. ASX Trade24 is ASX global trading platform for derivatives. It is globally distributed with network access points (gateways) located in Chicago, New York, London, Hong Kong, Singapore, Sydney and Melbourne. It also allows for true 24-hour trading, and simultaneously maintains two active trading days which enables products to be opened for trading in the new trading day in one time zone while products are still trading under the previous day. Opening times. The normal trading or business days of the ASX are week-days, Monday to Friday. ASX does not trade on national public holidays: New Year's Day (1 January), Australia Day (26 January, and observed on this day or the first business day after this date), Good Friday (that varies each year), Easter Monday, Anzac day (25 April), Queen's birthday (June), Christmas Day (25 December) and Boxing Day (26 December). On each trading day there is a pre-market session from 7:00 am to 10:00 am AEST and a normal trading session from 10:00 am to 4:00 pm AEST. The market opens alphabetically in single-price auctions, phased over the first ten minutes, with a small random time built in to prevent exact prediction of the first trades. There is also a single-price auction between 4:10 pm and 4:12 pm to set the daily closing prices. Settlement. Security holders hold shares in one of two forms, both of which operate as uncertificated holdings, rather than through the issue of physical share certificates: Clearing House Electronic Sub-register System (CHESS). The investor's controlling participant (normally a broker) sponsors the client into CHESS. The security holder is given a "holder identification number" (HIN) and monthly statements are sent to the security holder from the CHESS system when there is a movement in their holding that month. Issuer-sponsored. The company's share register administers the security holder's holding and issues the investor with a security-holder reference number (SRN) which may be quoted when selling. Holdings may be moved from issuer-sponsored to CHESS or between different brokers by electronic message initiated by the controlling participant. Short selling. Main article: Short (finance). Short selling of shares is permitted on the ASX, but only among designated stocks and with certain conditions: ASX trading participants (brokers) must report all daily gross short sales to ASX. The report will aggregate the gross short sales as reported by each trading participant at an individual stock level. ASX publishes aggregate gross short sales to ASX participants and the general public.[13]
Many brokers do not offer short selling to small private investors. LEPOs can serve as an equivalent, while contracts for difference (CFDs) offered by third-party providers are another alternative. In September 2008, ASIC suspended nearly all forms of short selling due to concerns about market stability in the ongoing global financial crisis.[14][15] The ban on covered short selling was lifted in May 2009.[16] Also, in the biggest change for ASX in 15 years, ASTC Settlement Rule 10.11.12 was introduced, which requires the broker to provide stocks when settlement is due, otherwise the broker must buy the stock on the market to cover the shortfall. The rule requires that if a Failed Settlement Shortfall exists on the second business day after the day on which the Rescheduled Batch Instruction was originally scheduled for settlement (that is, generally on T+5), the delivering settlement participant must either: close out the Failed Settlement Shortfall on the next business day by purchasing the number of Financial Products of the relevant class equal to the shortfall; or
acquire under a securities lending arrangement the number of Financial Products of the relevant class equal to the shortfall and deliver those Financial Products in Batch Settlement no more than two business days later.[17] Options. Options on leading shares are traded on the ASX, with standardised sets of strike prices and expiry dates. Liquidity is provided by market makers who are required to provide quotes. Each market maker is assigned two or more stocks. A stock can have more than one market maker, and they compete with one another. A market maker may choose one or both of: Make a market continuously, on a set of 18 options. Make a market in response to a quote request, in any option up to 9 months out. In both cases there is a minimum quantity (5 or 10 contracts depending on the shares) and a maximum spread permitted. Due to the higher risks in options, brokers must check clients' suitability before allowing them to trade options. Clients may both take (i.e. buy) and write (i.e. sell) options. For written positions, the client must put up margin. Interest rate market. The ASX interest rate market is the set of corporate bonds, floating rate notes, and bond-like preference shares listed on the exchange. These securities are traded and settled in the same way as ordinary shares, but the ASX provides information such as their maturity, effective interest rate, etc., to aid comparison.[18] Futures. The Sydney Futures Exchange (SFE) was the 10th largest derivatives exchange in the world, providing derivatives in interest rates, equities, currencies and commodities. The SFE is now part of ASX and its most active products are: SPI 200 Futures – Futures contracts on an index representing the largest 200 stocks on the Australian Stock Exchange by market capitalisation. AU 90-day Bank Accepted Bill Futures – Australia's equivalent of T-Bill futures. 3-Year Bond Futures – Futures contracts on Australian 3-year bonds. 10-Year Bond Futures – Futures contracts on Australian 10-year bonds.
The ASX trades futures over the ASX 50, ASX 200 and ASX property indexes, and over grain, electricity and wool. Options over grain futures are also traded. Market indices. The ASX maintains stock indexes concerning stocks traded on the exchange in conjunction with Standard & Poor's. There is a hierarchy of index groups called the S&P/ASX 20, S&P/ASX 50, S&P/ASX 100, S&P/ASX 200 and S&P/ASX 300, notionally containing the 20, 50, 100, 200 and 300 largest companies listed on the exchange, subject to some qualifications. Sharemarket Game. The ASX Sharemarket Game give members of the public and secondary school students the chance to learn about investing in the sharemarket using real market prices. Participants receive a hypothetical $50,000 to buy and sell shares in 150 companies and track the progress of their investments over the duration of the game.[19] Merger talks with SGX. ASX was (25 October 2010) in merger talks with Singapore Exchange (SGX). While there was an initial expectation that the merger would have created a bourse with a market value of US$14 billion,[20] this was a misconception; the final proposal intended that the ASX and SGX bourses would have continued functioning separately. The merger was blocked by Treasurer of Australia Wayne Swan on 8 April 2011, on advice from the Foreign Investment Review Board that the proposed merger was not in the best interests of Australia.[21]”


::::::::::::::::::::::::::::::::::::::::::::::::::

```python

tokens = simple_preprocess(text)
```

For Topic Modeling we need to map each word to a unique ID through creating a dictionary:

```python

dictionary = corpora.Dictionary([tokens])
```

And the created dictionary should be converted into a bag-of-words. We do that with doc2bow().

```python

corpus = [dictionary.doc2bow(tokens)]
```

Next, we use Latent Dirichlet Allocation (LDA) which is a popular topic modeling technique because it assumes documents are produced from a mixture of topics. These topics then generate words based on their probability distribution. Set up the LDA model with the number of topics and train it on the corpus: 

```python

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1)
```

Finally, we can print the topics and their word distributions from our text:

```python

topics = lda_model.print_topics(num_words=4)
    for topic in topics:
    print(topic)
```


::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: How does the topic modeling method help researchers? What about the text summarization? 
What are some of the challenges and limitations of text analysis in your research field (material science)? Consider some of the factors that affect the quality and accuracy of text analysis, such as data availability, language diversity, and domain specificity. How do these factors pose problems or difficulties for text analysis in material science?


:::::::::::::::::::::::::::::::::::::::::::::::





::::::::::::::::::::::::::::::::::::: challenge

### Chemistry Joke

Q: Use the Gensim library to perform topic modeling on the following text, print the original text and the list of topics and their keywords. 



:::::::::::::::::::::::::::::::::::::::::: spoiler
### Text

text = " Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers. Additionally, perovskite nanocrystals exhibit high photoluminescence efficiencies, meaning they can efficiently convert absorbed light into emitted light, further adding to their potential for various optoelectronic applications." 

::::::::::::::::::::::::::::::::::::::::::::::::::


You can use the following code to load the Gensim library and the pre-trained model for topic modeling:


```python

import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
```

:::::::::::::::: solution

A: 

```python

import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
tokens = simple_preprocess(text)
dictionary = corpora.Dictionary([tokens])
corpus = [dictionary.doc2bow(tokens)]
model = LdaModel(corpus, num_topics=2, id2word=dictionary)
print(text)
print(model.print_topics())


```

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



































































































::: callout
This is a callout block. It contains at least three colons
:::



:::::::::::::::::::::::::::::::::::::::::: spoiler

### What Else Might We Use A Spoiler For?


::::::::::::::::::::::::::::::::::::::::::::::::::





::::::::::::::::::::::::::::::::::::: challenge

### Chemistry Joke

Q: If you aren't part of the solution, then what are you?

:::::::::::::::: solution

A: part of the precipitate

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::








## Neural Networks

A **neural network** is an artificial intelligence technique loosely based on the way neurons in the brain work. 

### A single nueron
A neural network consists of connected computational units called **neurons**. Each neuron will:

- Take one or more inputs ($x_1, x_2, ...$), e.g., input data expressed as floating point numbers.
- Conduct three main operations most of the time:
    - Calculate the weighted sum of the inputs where ($w_1, w_2, ... $) indicate weights
    - Add an extra constant weight (i.e. a bias term) to this weighted sum
    - Apply a non-linear function to the output so far (using a predefined activation function such as the ReLU function)
- Return one output value, again a floating point number.

One example equation to calculate the output for a neuron is: $output=ReLU(∑i(xi∗wi)+bias)$

![](fig/03_neuron.png){alt='diagram of a single neuron taking multiple inputs and their associated weights in and then applying an activation function to predict a single output'}

### Combining multiple neurons into a network

Multiple neurons can be joined together by connecting the output of one to the input of another. These connections are associated with weights that determine the 'strength' of the connection, and the weights are adjusted during training. In this way, the combination of neurons and connections describe a computational graph, an example can be seen in the image below. 

In most neural networks neurons are aggregated into layers. Signals travel from the input layer to the output layer, possibly through one or more intermediate layers called hidden layers. The image below illustrates an example of a neural network with three layers, each circle is a neuron, each line is an edge and the arrows indicate the direction data moves in.

![The image above is by Glosser.ca, [CC BY-SA 3.0], via Wikimedia Commons, [original source]](fig/03_neural_net.png){alt='diagram of a neural with four neurons taking multiple inputs and their weights and predicting multiple outputs'}

Neural networks aren't a new technique, they have been around since the late 1940s. But until around 2010 neural networks tended to be quite small, consisting of only 10s or perhaps 100s of neurons. This limited them to only solving quite basic problems. Around 2010 improvements in computing power and the algorithms for training the networks made much larger and more powerful networks practical. These are known as deep neural networks or Deep Learning.

## Convolutional Neural Networks

A convolutional neural network (CNN) is a type of artificial neural network (ANN) most commonly applied to analyze visual imagery. They are designed to recognize the spatial structure of images when extracting features.

### Step 4. Build an architecture from scratch or choose a pretrained model

Let us explore how to build a neural network from scratch. Although this sounds like a daunting task, with Keras it is surprisingly straightforward. With Keras you compose a neural network by creating layers and linking them together.

### Parts of a neural network

There are three main components of a neural network:

- CNN Part 1. Input Layer
- CNN Part 2. Hidden Layers
- CNN Part 3. Output Layer

The output from each layer becomes the input to the next layer.

#### CNN Part 1. Input Layer

The Input in Keras gets special treatment when images are used. Keras automatically calculates the number of inputs and outputs a specific layer needs and therefore how many edges need to be created. This means we must let Keras know how big our input is going to be. We do this by instantiating a `keras.Input` class and pass it a tuple to indicate the dimensionality of the input data.

The input layer is created with the `tf.keras.Input` function and its first parameter is the expected shape of the input:

```
keras.Input(shape=None, batch_size=None, dtype=None, sparse=None, batch_shape=None, name=None, tensor=None)
```

In our case, the shape of an image is defined by its pixel dimensions and number of channels:

```python
# recall the shape of the images in our dataset
print(train_images.shape)
```
```output
(40000, 32, 32, 3) # number of images, image width in pixels, image height in pixels, number of channels (RGB)
```

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Create the input layer for our network

Hint 1: Specify shape argument only and use defaults for the rest.
Hint 2: The shape of our input dataset includes the total number of images. We want to take a slice of the shape for a single individual image to use an input.

```python
    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
   inputs_intro = keras.Input(#blank#)
```

:::::::::::::::::::::::: solution 

```output
    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
   inputs_intro = keras.Input(shape=train_images.shape[1:])
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


#### CNN Part 2. Hidden Layers

The next component consists of the so-called hidden layers of the network. The reason they are referred to as hidden is because the true values of their nodes are unknown.

In a CNN, the hidden layers typically consist of convolutional, pooling, reshaping (e.g., Flatten), and dense layers. 

Check out the [Layers API] section of the Keras documentation for each layer type and its parameters.


##### **Convolutional Layers**

A **convolutional** layer is a fundamental building block in a CNN designed for processing structured grid data, such as images. It applies convolution operations to input data using learnable filters or kernels, extracting local patterns and features (e.g. edges, corners). These filters enable the network to capture hierarchical representations of visual information, allowing for effective feature learning.

To find the particular features of an image, CNNs make use of a concept from image processing that precedes Deep Learning.

A **convolution matrix**, or **kernel**, is a matrix transformation that we 'slide' over the image to calculate features at each position of the image. For each pixel, we calculate the matrix product between the kernel and the pixel with its surroundings. Here is one example of a 3x3 kernel used to detect edges:

```
[[-1, -1, -1],
 [0,   0,  0]
 [1,   1,  1]]
```
This kernel will give a high value to a pixel if it is on a horizontal border between dark and light areas.

In the following image, the effect of such a kernel on the values of a single-channel image stands out. The red cell in the output matrix is the result of multiplying and summing the values of the red square in the input, and the kernel. Applying this kernel to a real image demonstrates it does indeed detect horizontal edges.

![](fig/03_conv_matrix.png){alt='6x5 input matrix representing a single colour channel image being multipled by a 3x3 kernel to produce a 4x4 output matrix to detect horizonal edges in an image '}

![](fig/03_conv_image.png){alt='single colour channel image of a cat multiplied by a 3x3 kernel to produce an image of a cat where the edges  stand out'}

There are several types of convolutional layers available in Keras depending on your application. We use the two-dimensional layer typically used for images:

```
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="valid", activation=None, **kwargs)
```

We want to create a Conv2D layer with 16 filters, a 3x3 kernel size, and the 'relu' activation function.

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Create a 2D convolutional layer for our network

Hint 1: The input to each layer is the output of the previous layer.

```python
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(filters=#blank#, kernel_size=#blank#, activation=#blank#)(#blank#)
```

:::::::::::::::::::::::: solution 

```output
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs_intro)
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


The instantiation here has three parameters and a seemingly strange combination of parentheses, so let us break it down.

- The first parameter is the number of filters in this layer. This is one of the hyperparameters of our system and should be chosen carefully.
    - Good practice is to start with a relatively small number of filters in the first layer to prevent overfitting.
    - Choosing a number of filters as a power of two (e.g., 32, 64, 128) is common.
- The second parameter is the kernel size which we already discussed. Smaller kernels are often used to capture fine-grained features and odd-sized filters are preferred because they have a centre pixel which helps maintain spatial symmetry during convolutions.
- The third parameter is the activation function to use.
    - Here we choose **relu** which is one of the most commonly used in deep neural networks that is proven to work well. 
    - We will discuss activation functions later in **Step 9. Tune hyperparameters** but to satisfy your curiosity, `ReLU` stands for Rectified Linear Unit (ReLU).
- Next is an extra set of parenthenses with inputs in them that means after an instance of the Conv2D layer is created, it can be called as if it was a function. This tells the Conv2D layer to connect the layer passed as a parameter, in this case the inputs.
- Finally, we store a reference so we can pass it to the next layer.


:::::::::::::::::::::::::::::::::::::: callout

## Playing with convolutions

Convolutions applied to images can be hard to grasp at first. Fortunately, there are resources out there that enable users to interactively play around with images and convolutions:

- [Image kernels explained] illustrates how different convolutions can achieve certain effects on an image, like sharpening and blurring.

- The [convolutional neural network cheat sheet] provides animated examples of the different components of convolutional neural nets.
:::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

## CHALLENGE Border pixels

What do you think happens to the border pixels when applying a convolution?

:::::::::::::::::::::::: solution

There are different ways of dealing with border pixels. 

- You can ignore them, which means your output image is slightly smaller then your input. 
- It is also possible to 'pad' the borders, e.g., with the same value or with zeros, so that the convolution can also be applied to the border pixels. In that case, the output image will have the same size as the input image.
:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


##### **Pooling Layers**

The convolutional layers are often intertwined with **Pooling** layers. As opposed to the convolutional layer used in feature extraction, the pooling layer alters the dimensions of the image and reduces it by a scaling factor effectively decreasing the resolution of your picture. 

The rationale behind this is that higher layers of the network should focus on higher-level features of the image. By introducing a pooling layer, the subsequent convolutional layer has a broader 'view' on the original image.

Similar to convolutional layers, Keras offers several pooling layers and one used for images (2D spatial data):

```
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None, name=None, **kwargs)
```

We want to create a pooling layer with input window sized 2,2.

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Create a Pooling layer for our network

Hint 1: The input to each layer is the output of the previous layer.

```python
    # Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D(#blank#)(#blank#)
```

:::::::::::::::::::::::: solution 

```output
    # Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D(pool_size=(2, 2))(x_intro)
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

The instantiation here has a single parameter, pool_size.

The function downsamples the input along its spatial dimensions (height and width) by taking the **maximum** value over an input window (of size defined by pool_size) for each channel of the input. By taking the maximum instead of the average, the most prominent features in the window are emphasized.

A 2x2 pooling size reduces the width and height of the input by a factor of 2. Empirically, a 2x2 pooling size has been found to work well in various for image classification tasks and also strikes a balance between down-sampling for computational efficiency and retaining important spatial information.


##### **Dense layers**

A **dense** layer has a number of neurons, which is a parameter you choose when you create the layer. When connecting the layer to its input and output layers every neuron in the dense layer gets an edge (i.e. connection) to **all** of the input neurons and **all** of the output neurons.

![](fig/03-neural_network_sketch_dense.png){alt='diagram of a neural network with multiple inputs feeding into to two seperate dense layers with connections between all the inputs and outputs'}

This layer is called fully connected, because all input neurons are taken into account by each output neuron. It aggregates global information about the features learned in previous layers to make a decision about the class of the input.

In Keras, a densely-connected layer is defined:

```
keras.layers.Dense(units, activation=None, **kwargs)
```

Units in this case refer to the number of neurons.

The choice of how many neurons to specify is often determined through experimentation and can impact the performance of our CNN. Too few neurons may not capture complex patterns in the data but too many neurons may lead to overfitting.

We will choose 64 for our dense layer and 'relu' activation.

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Create a Dense layer for our network

Hint 1: The input to each layer is the output of the previous layer.

```python
    # Dense layer with 64 neurons and ReLU activation
    x_intro = keras.layers.Dense(units=#hidden#, activation=#hidden#)(#hidden#)
```

:::::::::::::::::::::::: solution 

```output
    # Dense layer with 64 neurons and ReLU activation
    x_intro = keras.layers.Dense(64, activation='relu')(x_intro)
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



##### **Reshaping Layers: Flatten**

The next type of hidden layer used in our introductory model is a type of reshaping layer defined in Keras by the `tf.keras.layers.Flatten` class. It is necessary when transitioning from convolutional and pooling layers to fully connected layers.

```
keras.layers.Flatten(data_format=None, **kwargs)
```

The **Flatten** layer converts the output of the previous layer into a single one-dimensional vector that can be used as input for a dense layer.

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Create a Flatten layer for our network

Hint 1: The input to each layer is the output of the previous layer.

```python
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_intro = keras.layers.Flatten()(=#hidden#)
```

:::::::::::::::::::::::: solution 

```output
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_intro = keras.layers.Flatten()(x_intro)
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::: spoiler

#### What does **Flatten** mean exactly?

A flatten layer function is typically used to transform the two-dimensional arrays (matrices) generated by the convolutional and pooling layers into a one-dimensional array. This is necessary when transitioning from the convolutional/pooling layers to the fully connected layers, which require one-dimensional input.

During the convolutional and pooling operations, a neural network extracts features from the input images, resulting in multiple feature maps, each represented by a matrix. These feature maps capture different aspects of the input image, such as edges, textures, or patterns. However, to feed these features into a fully connected layer for classification or regression tasks, they must be a single vector.

The flatten layer takes each element from the feature maps and arranges them into a single long vector, concatenating them along a single dimension. This transformation preserves the spatial relationships between the features in the original image while providing a suitable format for the fully connected layers to process.

:::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::: callout

## Is one layer of each type enough?

Not for complex data! 

A typical architecture for image classification is likely to include at least one convolutional layer, one pooling layer, one or more dense layers, and possibly a flatten layer.

Convolutional and Pooling layers are often used together in multipe sets to capture a wider range of features and learn more complex representations of the input data. Using this technique, the network can learn a hierarchical representation of features, where simple features detected in early layers are combined to form more complex features in deeper layers.

There isn't a strict rule of thumb for the number of sets of convolutional and pooling layers to start with, however, there are some guidelines.

We are starting with a relatively small and simple architecture because we are limited in time and computational resources. A simple CNN with one or two sets of convolutional and pooling layers can still achieve decent results for many tasks but for your network you will experiment with different architectures.

:::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Using the four layers above, create a hidden layer architecture that contains:

- 2 sets of Conv2D and Pooling layers, with 16 and 32 filters respectively
- 1 Flatten layer
- 1 Dense layer with

Hint 1: The input to each layer is the output of the previous layer.

:::::::::::::::::::::::: solution 

```output
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_intro)
    # Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_intro)
    # Second Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_intro = keras.layers.Flatten()(x_intro)
    # Dense layer with 64 neurons and ReLU activation
    x_intro = keras.layers.Dense(64, activation='relu')(x_intro)
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


#### CNN Part 3. Output Layer

Recall for the outputs we asked ourselves what we want to identify from the data. If we are performing a classification problem, then typically we have one output for each potential class. 

In traditional CNN architectures, a dense layer is typically used as the final layer for classification. This dense layer receives the flattened feature maps from the preceding convolutional and pooling layers and outputs the final class probabilities or regression values.

For multiclass data, the `softmax` activation is used instead of `relu` because it helps the computer give each option (class) a likelihood score, and the scores add up to 100 per cent. This way, it's easier to pick the one the computer thinks is most probable.

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Create an Output layer for our network using a Dense layer

Hint 1: The input to each layer is the output of the previous layer.
Hint 2: The units (neurons) should be the same as number of classes as our dataset.
Hint 3: Use softmax activation.

```python
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_intro = keras.layers.Dense(units=#hidden#, activation=#hidden#)(#hidden#)
```

:::::::::::::::::::::::: solution 

```output
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro)
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


## Putting it all together

::::::::::::::::::::::::::::::::::::: challenge 

## CHALLENGE Create a function that defines a CNN using the input, hidden, and output layers in previous challenges.

Hint 1: The input to each layer is the output of the previous layer.
Hint 2: The units (neurons) should be the same as number of classes as our dataset.
Hint 3: Use softmax activation.

:::::::::::::::::::::::: solution 

```output
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro)
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

```python
def create_model_intro():
    
    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
    inputs_intro = keras.Input(shape=train_images.shape[1:])
    
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_intro)
    # Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_intro)
    # Second Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_intro = keras.layers.Flatten()(x_intro)
    # Dense layer with 64 neurons and ReLU activation
    x_intro = keras.layers.Dense(64, activation='relu')(x_intro)
    
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro)
    
    # create the model
    model_intro = keras.Model(inputs = inputs_intro, 
                              outputs = outputs_intro, 
                              name = "cifar_model_intro")
    
    return model_intro
```

Use the function you created to create the introduction model and view a summary of it's structure.

```python
# create the introduction model
model_intro = create_model_intro()

# view model summary
model_intro.summary()
```
```output
Model: "cifar_model_intro"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 32, 32, 3)]       0         
                                                                 
 conv2d (Conv2D)             (None, 30, 30, 16)        448       
                                                                 
 max_pooling2d (MaxPooling2  (None, 15, 15, 16)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 6, 6, 32)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 64)                73792     
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 79530 (310.66 KB)
Trainable params: 79530 (310.66 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

:::::::::::::::::::::::::::::::::::::: callout

## How to choose an architecture?

Even for this neural network, we had to make a choice on the number of hidden neurons. Other choices to be made are the number of layers and type of layers. You might wonder how you should make these architectural choices. Unfortunately, there are no clear rules to follow here, and it often boils down to a lot of trial and error. However, it is recommended to explore what others have done with similar datasets and problems. Another best practice is to start with a relatively simple architecture. Once running start to add layers and tweak the network to test if performance increases. 

::::::::::::::::::::::::::::::::::::::::::::::

## We have a model now what?

This CNN should be able to run with the CIFAR-10 dataset and provide reasonable results for basic classification tasks. However, do keep in mind this model is relatively simple, and its performance may not be as high as more complex architectures. The reason it's called deep learning is because in most cases, the more layers we have, i.e. the deeper and more sophisticated CNN architecture we use, the better the performance.

How can we tell? We can inspect a couple metrics produced during the training process to detect whether our model is underfitting or overfitting. To do that, we continue with the next steps in our Deep Learning workflow, **Step 5. Choose a loss function and optimizer** and **Step 6. Train model**. 


::::::::::::::::::::::::::::::::::::: keypoints 

- Artificial neural networks (ANN) are a machine learning technique based on a model inspired by groups of neurons in the brain.
- Convolution neural networks (CNN) are a type of ANN designed for image classification and object detection.
- The number of filters corresponds to the number of distinct features the layer is learning to recognise whereas the kernel size determines the level of features being captured.
- A CNN can consist of many types of layers including convolutional, pooling, flatten, and dense (fully connected) layers
- Convolutional layers are responsible for learning features from the input data.
- Pooling layers are often used to reduce the spatial dimensions of the data.
- The flatten layer is used to convert the multi-dimensional output of the convolutional and pooling layers into a flat vector.
- Dense layers are responsible for combining features learned by the previous layers to perform the final classification.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->

[CC BY-SA 3.0]: https://creativecommons.org/licenses/by-sa/3.0
[original source]: https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg
[Layers API]: https://keras.io/api/layers/
[Image kernels explained]: https://setosa.io/ev/image-kernels/
[convolutional neural network cheat sheet]: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

