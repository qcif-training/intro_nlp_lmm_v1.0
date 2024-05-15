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

nlp = spacy.load("en_core_web_sm")

```

Create a variable to store your text and then apply the model to process your text (text from [Wikipedia](https://en.wikipedia.org/wiki/Australian_Securities_Exchange)):

:::::::::::::::::::::::::::::::::::::::::: spoiler

### Text

text = “Australian Shares Exchange Ltd (ASX) is an Australian public company that operates Australia's primary shares exchange, the Australian Shares Exchange (sometimes referred to outside of Australia as, or confused within Australia as, The Sydney Stock Exchange, a separate entity). The ASX was formed on 1 April 1987, through incorporation under legislation of the Australian Parliament as an amalgamation of the six state securities exchanges, and merged with the Sydney Futures Exchange in 2006. Today, ASX has an average daily turnover of A$4.685 billion and a market capitalization of around A$1.6 trillion, making it one of the world's top 20 listed exchange groups, and the largest in the southern hemisphere. ASX Clear is the clearing house for all shares, structured products, warrants and, ASX Equity Derivatives.”

::::::::::::::::::::::::::::::::::::::::::::::::::


Use for loop to print all the named entities in the document:


```python

doc = nlp(text)

For ent in doc.ents:
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

A: You can use the following code to get information about each one of the labels. For example, we want to know what GPE represents here. We can use *explain()* to get the required information:
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
Download the necessary NLTK resources and import the required toolkit:


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

### Challenge

Teamwork: To better understand this and to find the connection between concepts we have learned so far, let’s match the following terms to their brief definitions:

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/2be22cfb-4ed8-4f65-85bf-31962a40835f)

:::::::::::::::: solution

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/d7d9a9a4-c2cb-4f37-bc6c-7b2f99133fe3)

:::::::::::::::::::::::::
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

### Challenge

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




::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: Use the genism to perform topic modeling on the following two different texts and provide a comparison.


:::::::::::::::::::::::::::::::::::::::::: spoiler
### Text

text1 = "Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers. Additionally, perovskite nanocrystals exhibit high photoluminescence efficiencies, meaning they can efficiently convert absorbed light into emitted light, further adding to their potential for various optoelectronic applications." 


text2 = "Graphene is a one-atom-thick sheet of carbon atoms arranged in a honeycomb lattice. It is a remarkable material with unique properties, including high electrical conductivity, thermal conductivity, mechanical strength, and optical transparency. Graphene has the potential to revolutionize various fields, including electronics, photonics, and composite materials. Due to its excellent electrical conductivity, graphene is a promising candidate for next-generation electronic devices, such as transistors and sensors. Additionally, its high thermal conductivity makes it suitable for heat dissipation applications."

::::::::::::::::::::::::::::::::::::::::::::::::::



:::::::::::::::: solution

A: After storing the two texts in text1 and text2, preprocess the text (e.g., tokenization, stop word removal, stemming/lemmatization). Split the texts into documents:


```python

import gensim
from gensim import corpora
documents = [text1.split(), text2.split()]

# Create a dictionary:
dictionary = corpora.Dictionary(documents)

# Create a corpus (bag of words representation):
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Train the LDA model (adjust num_topics as needed):
lda_model = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=3, passes=20)

# Print the original texts:
print("Original Texts:")
print(f"Text 1:\n{text1}\n")
print(f"Text 2:\n{text2}\n")

# Identify shared and distinct keywords for each topic:
print("Topics and Keywords:")
for topic in lda_model.show_topics(formatted=False):
    print(f"\nTopic {topic[0]}:")
    topic_words = [w[0] for w in topic[1]]
    print(f"Text 1 Keywords:", [w for w in topic_words if w in text1])
    print(f"Text 2 Keywords:", [w for w in topic_words if w in text2])

# Explain the conceptual similarity:
print("\nConceptual Similarity:")
print("Both texts discuss novel materials (perovskite nanocrystals and graphene) with unique properties. While the specific applications and functionalities differ slightly, they both highlight the potential of these materials for various technological advancements.")
```

:::::::::::::::::::::::::::::::::::::::::: spoiler
### Output

Original Texts:

Text 1:
Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers. Additionally, perovskite nanocrystals exhibit high photoluminescence efficiencies, meaning they can efficiently convert absorbed light into emitted light, further adding to their potential for various optoelectronic applications.

Text 2:
Graphene is a one-atom-thick sheet of carbon atoms arranged in a honeycomb lattice. It is a remarkable material with unique properties, including high electrical conductivity, thermal conductivity, mechanical strength, and optical transparency. Graphene has the potential to revolutionize various fields, including electronics, photonics, and composite materials. Due to its excellent electrical conductivity, graphene is a promising candidate for next-generation electronic devices, such as transistors and sensors. Additionally, its high thermal conductivity makes it suitable for heat dissipation applications.


Topics and Keywords:

```
Topic 0:

Text 1 Keywords: ['applications', 'devices', 'material', 'optoelectronic', 'properties']
Text 2 Keywords: ['applications', 'conductivity', 'electronic', 'graphene', 'material', 'potential']

Topic 1:
	
Text 1 Keywords: ['bandgaps', 'crystal', 'electronic', 'properties', 'structure']
Text 2 Keywords: ['conductivity', 'electrical', 'graphene', 'material', 'properties']

Topic 2:

Text 1 Keywords: ['absorption', 'emit', 'light', 'spectrum']
Text 2 Keywords: ['conductivity', 'graphene', 'material', 'optical', 'potential']
```

Conceptual Similarity:

Both texts discuss novel materials (perovskite nanocrystals and graphene) with unique properties. While the specific applications and functionalities differ slightly (optoelectronic devices vs. electronic devices), they both highlight the potential of these materials for various technological advancements. Notably, both topics identify "material," "properties," and "applications" as keywords, suggesting a shared focus on the materials' characteristics and their potential uses. Additionally, keywords like "electronic," "conductivity," and "potential" appear in both texts within different topics, indicating a conceptual overlap in exploring the electronic properties and potential applications of these materials.


::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::




::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: Use the Gensim library to perform topic modeling on the following text print the original text and the list of topics and their keywords. 

:::::::::::::::::::::::::::::::::::::::::: spoiler
### Text

text = "Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation." 

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

```

output = Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.

[(0, '0.051*"natural" + 0.051*"language" + 0.051*"processing" + 0.027*"nlp" + 0.027*"challenges" + 0.027*"speech" + 0.027*"recognition" + 0.027*"understanding" + 0.027*"generation" + 0.027*"frequently"'), (1, '0.051*"natural" + 0.051*"language" + 0.051*"computers" + 0.027*"interactions" + 0.027*"between" + 0.027*"human" + 0.027*"languages" + 0.027*"particular" + 0.027*"program" + 0.027*"process"')]


```

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



::: callout

### Challenge of using small size corpus

The warning message "too few updates, training might not converge") arises when you are using a very small corpus for topic modeling with Latent Dirichlet Allocation (LDA) in Gensim.
LDA relies on statistical analysis of documents to discover hidden topics. With a limited corpus (one document in your case), there aren't enough data points for the model to learn robust topics. Increasing the number of documents (corpus size) generally improves the accuracy and convergence of LDA models.

:::


## 3.3. Text Summarization


Text summarization in NLP is the process of creating a concise and coherent version of a longer text document, preserving its key information. There are two primary approaches to text summarization:

1. Extractive Summarization: This method involves identifying and extracting key sentences or phrases directly from the original text to form the summary. It is akin to creating a highlight reel of the most important points.
2. Abstractive Summarization: This approach goes beyond mere extraction; it involves understanding the main ideas and then generating new, concise text that captures the essence of the original content. It is similar to writing a synopsis or an abstract for a research paper.


In the next part of the workshop, we will explore advanced tools like transformers, which can generate summaries that are more coherent and closer to what a human might write. Transformers use models like BERT and GPT to understand the context and semantics of the text, allowing for more sophisticated abstractive summaries.




::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: Fill in the blanks with the correct terms related to text summarization:


- ------ summarization selects sentences directly from the original text, while ------ summarization generates new sentences.
- ------ are advanced tools used for generating more coherent and human-like summaries.
- The ------ and ------ models are examples of transformers that understand the context and semantics of the text.
- ------ summarization can often create summaries that are more ------ and coherent than ------ methods.
- Advanced summarization tools use ------ and ------ to interpret and condense text.


:::::::::::::::: solution

A: 

- Extractive summarization selects sentences directly from the original text, while abstractive summarization generates new sentences.
  
- Transformers are advanced tools used for generating more coherent and human-like summaries.
  
- The BERT and GPT models are examples of transformers that understand the context and semantics of the text.
  
- Abstractive summarization can often create summaries that are more concise and coherent than extractive methods.
  
- Advanced summarization tools use machine learning and natural language processing to interpret and condense text.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



::: callout

In the rapidly evolving field of NLP, summarization tasks are increasingly being carried out using transformer-based models due to their advanced capabilities in understanding context and generating coherent summaries. Tools like Gensim’s summarization module 

```python
from gensim.summarization import summarize
```

have become outdated and were removed in its 4.0 release ([source](https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4#12-removed-gensimsummarization)), as they relied on extractive methods that simply selected parts of the existing text, which is less effective compared to the abstractive approach of transformers. These cutting-edge transformer models, which can create concise and fluent summaries by generating new sentences, are leading to the gradual disappearance of older, less efficient summarization methods.

:::



::::::::::::::::::::::::::::::::::::: keypoints 

- Named Entity Recognition (NER) is crucial for identifying and categorizing key information in text, such as names of people, organizations, and locations.
- Topic Modeling helps uncover the underlying thematic structure in a large corpus of text, which is beneficial for summarizing and understanding large datasets.
- Text Summarization provides a concise version of a longer text, highlighting the main points, which is essential for quick comprehension of extensive research material.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->

