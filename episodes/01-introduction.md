---
title: "Introduction to Natural Language Processing"
teaching: 10
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are some common research applications of NLP?
- What are the basic concepts and terminology of NLP?
- How can I use NLP in my research field?
- How can I acquire data for NLP tasks?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Define natural language processing and its goals.
- Identify the main research applications and challenges of NLP.
- Explain the basic concepts and terminology of NLP, such as tokens, lemmas, and n-grams.
- Use some popular datasets and libraries to acquire data for NLP tasks.

::::::::::::::::::::::::::::::::::::::::::::::::


## 1.1. Introduction to NLP Workshop

Natural Language Processing (NLP) is becoming a popular and robust tool for a wide range of research projects. In this episode we embark on a journey to explore the transformative power of NLP tools in the realm of research. 

It is tailored for researchers who are keen on harnessing the capabilities of NLP to enhance and expedite their work. Whether you are delving into text classification, extracting pivotal information, discerning sentiments, summarizing extensive documents, translating across languages, or developing sophisticated question-answering systems, this session will lay the foundational knowledge you need to leverage NLP effectively.

We will begin by delving into the **Common Applications of NLP in Research**, showcasing how these tools are not just theoretical concepts but practical instruments that drive forward today’s innovative research projects. From analyzing public sentiment to extracting critical data from a plethora of documents, NLP stands as a pillar in the modern researcher’s toolkit. 



![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/3d548b7f-cfd3-41d6-87bd-42f437d1415b)
[*DALL-E 3*]



Next, we’ll demystify the **Basic Concepts and Terminology of NLP**. Understanding these fundamental terms is crucial, as they form the building blocks of any NLP application. We’ll cover everything from the basics of a corpus to the intricacies of transformers, ensuring you have a solid grasp of the language used in NLP. 

Finally, we’ll guide you through Data Acquisition: Dataset Libraries, where you’ll learn about the treasure troves of data available at your fingertips. We’ll compare different libraries and demonstrate how to access and utilize these resources through hands-on examples. 

By the end of this episode, you will not only understand the significance of NLP in research but also be equipped with the knowledge to start applying these tools to your own projects. Prepare to unlock new potentials and streamline your research process with the power of NLP!


::::::::::::::::::::::::::::::::::::: challenge

## Discussion

Teamwork: What are some examples of NLP in your everyday life? Think of some situations where you interact with or use NLPs, such as online search, voice assistants, social media, etc. How do these examples demonstrate the usefulness of NLP in research projects?

:::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: challenge

## Discussion

Teamwork: What are some of examples of NLP in your daily research tasks? What are challenges of NLP that make it difficult, complex, and/or inaccurate?

:::::::::::::::::::::::::::::::::::::::::::::::


## 1.2.	Common Applications of NLP in Research

**Sentiment Analysis** is a powerful tool for researchers, especially in fields like market research, political science, and public health. It involves the computational identification of opinions expressed in text, categorizing them as positive, negative, or neutral. In market research, for instance, sentiment analysis can be applied to product reviews to gauge consumer satisfaction: a study could analyze thousands of online reviews for a new smartphone model to determine the overall public sentiment. This can help companies identify areas of improvement or features that are well-received by consumers. 



**Information Extraction** is crucial for quickly gathering specific information from large datasets. It is used extensively in legal research, medical research, and scientific studies to extract entities and relationships from texts. In legal research, for example, information extraction can be used to sift through case law to find precedents related to a particular legal issue. A researcher could use NLP to extract instances of “negligence” from thousands of case files, aiding in the preparation of legal arguments. 



**Text Summarization** helps researchers by providing concise summaries of lengthy documents, such as research papers or reports, allowing them to quickly understand the main points without reading the entire text. In biomedical research, text summarization can assist in literature reviews by providing summaries of research articles. For example, a researcher could use an NLP model to summarize articles on gene therapy, enabling them to quickly assimilate key findings from a vast array of publications.



**Topic Modeling** is used to uncover latent topics within large volumes of text, which is particularly useful in fields like sociology and history to identify trends and patterns in historical documents or social media data. 


![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/3d6bc7c3-dff0-430f-a44b-544d0944e59c)
[source](https://dl.acm.org/doi/10.1145/2133806.2133826)

For example, in historical research, topic modeling can reveal prevalent themes in primary source documents from a particular era. A historian might use NLP to analyze newspapers from the early 20th century to study public discourse around significant events like World War I.


**Named Entity Recognition** is a process where an algorithm takes a string of text (sentence or paragraph) and identifies relevant nouns (people, places, and organizations) that are mentioned in that string. NER is used in many fields in NLP, and it can help answer many real-world questions, such as: Which companies were mentioned in the news article? Were specified products mentioned in complaints or reviews? Does the tweet (recently rebranded to X) contain the name of a person? Does the tweet contain this person’s location?


![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/27b2f4a2-f0a8-4061-8f52-223f2a0f8e31)
[source](https://www.turing.com/kb/a-comprehensive-guide-to-named-entity-recognition)




::: callout
### Challenges of NLP

One of the significant challenges in NLP is dealing with the ambiguity of language. Words or phrases can have multiple meanings, and determining the correct one based on context can be difficult for NLP systems. In a research paper discussing “bank erosion,” an NLP system might confuse “bank” with a financial institution rather than the geographical feature, leading to incorrect analysis. 

**This challenge leads to the fact that the classical NLP systems often struggle with contextual understanding which is crucial in text analysis tasks.** This can lead to misinterpretation of the meaning and sentiment of the text. If a research paper mentions “novel results,” an NLP system might interpret “novel” as a literary work instead of “new” or “original,” which could mislead the analysis of the paper’s contributions.

:::



:::::::::::::::::::::::::::::::::::::::::: spoiler

### Suggested Resources:

- Python’s Natural Language Toolkit (NLTK) for sentiment analysis
- TextBlob, a library for processing textual data
- Stanford NER for named entity recognition
- spaCy, an open-source software library for advanced NLP
- Sumy, a Python library for automatic summarization of text documents
- BERT-based models for extractive and abstractive summarization
- Gensim for topic modeling and document similarity analysis
- MALLET, a Java-based package for statistical natural language processing

::::::::::::::::::::::::::::::::::::::::::::::::::





## 1.3. Basic Concepts and Terminology of NLP


::::::::::::::::::::::::::::::::::::: challenge

## Discussion

Teamwork: What are some of the basic concepts and terminology of natural language processing that you are familiar with or want to learn more about? Share your knowledge or questions with a partner or a small group, and try to explain or understand some of the key terms of natural language processing, such as tokens, lemmas, n-grams, etc.

:::::::::::::::::::::::::::::::::::::::::::::::


**Corpus**: A corpus is a collection of written texts, especially the entire works of a particular author or a body of writing on a particular subject. In NLP, a corpus is used as a large and structured set of texts that can be used to perform statistical analysis and hypothesis testing, check occurrences, or validate linguistic rules within a specific language territory.

**Token and Tokenization**: Tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing such as parsing or text mining. Tokenization is useful in situations where certain characters or words need to be treated as a single entity, despite any spaces or punctuation that might separate them.

**Stemming**: Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base, or root form—generally a written word form. The idea is to remove affixes to get to the root form of the word. Stemming is often used in search engines for indexing words. Instead of storing all forms of a word, a search engine can store only the stems, greatly reducing the size of the index while increasing retrieval accuracy.


![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/42718630-80a0-483f-81a7-61be10f5aa99)
[source](https://nirajbhoi.medium.com/stemming-vs-lemmatization-in-nlp-efc280d4e845)


**Lemmatization**: Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language. In Lemmatization, the root word is called Lemma. A lemma is the canonical form, dictionary form, or citation form of a set of words. For example, runs, running, and running are all forms of the word run, therefore run is the lemma of all these words.



**Part-of-Speech (PoS) Tagging**: Part-of-speech tagging is the process of marking up a word in a text as corresponding to a particular part of speech, based on both its definition and its context. It is a necessary step before performing more complex NLP tasks like parsing or grammar checking.



**Named Entity Recognition (NER)**: Named Entity Recognition is a process where an algorithm takes a string of text (sentence or paragraph) and identifies relevant nouns (people, places, and organizations) that are mentioned in that string. NER is used in many fields in NLP, and it can help answer many real-world questions, such as: Which companies were mentioned in the news article? Were specified products mentioned in complaints or reviews? Does the tweet (recently rebranded to X) contain the name of a person? Does the tweet contain this person’s location?



**Dependency Parsing**: Dependency parsing is the process of analyzing the grammatical structure of a sentence based on the dependencies between the words in a sentence. A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between “head” words and words that modify those heads.



**Chunking**: Chunking is a process of extracting phrases from unstructured text. Instead of just simple tokens that may not represent the actual meaning of the text, it’s also interested in extracting entities like noun phrases, verb phrases, etc. It’s basically a meaningful grouping of words or tokens.



**Word Embeddings**: Word embeddings are a type of word representation that allows words with similar meanings to have a similar representation. They are a distributed representation of text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems.



**Transformers**: Transformers are models that handle the ordering of words and other elements in a language. They are designed to handle sequential data, such as natural language, for tasks such as translation and text summarization. They are the foundation of most recent advances in NLP, including models like BERT and GPT.




## 1.4. Data Acquisition: Dataset Libraries:


Different data libraries offer various datasets that are useful for training and testing NLP models. These libraries provide access to a wide range of text data, from literary works to social media posts, which can be used for tasks such as sentiment analysis, topic modeling, and more.

**Natural Language Toolkit (NLTK)**: NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text-processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

**spaCy**: spaCy is a free, open-source library for advanced Natural Language Processing in Python. It’s designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems or to pre-process text for deep learning.

**Gensim**: Gensim is a Python library for topic modeling, document indexing, and similarity retrieval with large corpora. Targeted at the NLP and information retrieval communities, Gensim is designed to handle large text collections using data streaming and incremental online algorithms, which differentiates it from most other machine learning software packages that only target batch and in-memory processing.

**Hugging Face’s datasets**: This library provides a simple command-line interface to download and pre-process any of the major public datasets (text datasets in 467 languages and dialects, etc.) provided on the HuggingFace Datasets Hub. It’s designed to let the community easily add and share new datasets. Hugging Face Datasets simplifies working with data for machine learning, especially NLP tasks. It provides a central hub to find and load pre-processed datasets, saving you time on data preparation. You can explore a vast collection of datasets for various tasks and easily integrate them with other Hugging Face libraries.


Data acquisition using Hugging Face datasets library. First, we start with installing the library:

```python
pip install datasets

# To import the dataset, we can write:
from datasets import load_dataset
```

Use load_dataset with the dataset identifier in quotes. For example, to load the SQuAD question answering dataset:


```python
squad_dataset = load_dataset("squad")
# Use the info attribute to view the dataset description:
print(squad_dataset.info)
```

Each data point is a dictionary with keys corresponding to data elements (e.g., question, context, answer). Access them using those keys within square brackets:


```python
question = squad_dataset["train"][0]["question"]
context = squad_dataset["train"][0]["context"]
answer = squad_dataset["train"][0]["answer"]
```

We can use print() function to see the output:


```python
print(f"Question: {question}")
print(f"Context: {context}")
print(f"Answer: {answer}")
```



::::::::::::::::::::::::::::::::::::: challenge

## Challenge:

Q: Use the nltk library to acquire data for natural language processing tasks. You can use the following code to load the nltk library and download some popular datasets:


```python

import nltk
nltk.download()
```

Choose one of the datasets from the nltk downloader, such as brown, reuters, or gutenberg, and load it using the nltk.corpus module. Then, print the name, size, and description of the dataset. 


:::::::::::::::: solution


A: You can use the following code to access the dataset information:
Use the nltk library to acquire data for NLP tasks. Import the necessary libraries:


```python

import nltk
from nltk.corpus import gutenberg, brown

# Download the required data:
nltk.download('gutenberg')
nltk.download('brown')

print(gutenberg.readme())

# Access the downloaded data:
gutenberg_text = gutenberg.raw('austen-emma.txt')
brown_text = brown.words()


:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::























::::::::::::::::::::::::::::::::::::: challenge

## Chemistry Joke

Q: If you aren't part of the solution, then what are you?

:::::::::::::::: solution

A: part of the precipitate

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::




:::::::::::::::::::::::::::::::::::::::::: spoiler

### What Else Might We Use A Spoiler For?



::::::::::::::::::::::::::::::::::::::::::::::::::




::: callout
This is a callout block. It contains at least three colons
:::






::::::::::::::::::::::::::::::::::::: keypoints 

- Image datasets can be found online or created uniquely for your research question.
- Images consist of pixels arranged in a particular order.
- Image data is usually preprocessed before use in a CNN for efficiency, consistency, and robustness.
- Input data generally consists of three sets: a training set used to fit model parameters; a validation set used to evaluate the model fit on training data; and a test set used to evaluate the final model performance.

::::::::::::::::::::::::::::::::::::::::::::::::
