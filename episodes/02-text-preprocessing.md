---
title: 'Introduction to Text Preprocessing'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can I prepare data for NLP text analysis?
- How can I use spaCy for text preprocessing?


::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Define text preprocessing and its purpose for NLP tasks.
- Perform sentence segmentation, tokenization lemmatization, and stop-words removal, using spaCy.

::::::::::::::::::::::::::::::::::::::::::::::::



Text preprocessing is the method of cleaning and preparing text data for use in NLP. This step is vital because it transforms raw data into a format that can be analyzed and used effectively by NLP algorithms.



## 2.1. Sentence Segmentation

Sentence segmentation divides a text into its constituent sentences, which is essential for understanding the structure and flow of the content. We start with a field-specific text example and see how it works. We can start with a paragraph about perovskite nanocrystals from the context of material engineering. Divide it into sentences.

We can use the open-source library, spaCy, to perform this task. First, we import the spaCy library:

```python

import spacy
```

Then we need to Load the English language model:

```python
nlp = spacy.load("en_core_web_sm")
```

We can store our text here:

```python
perovskite_text = "Perovskite nanocrystals are a class of semiconductor nanocrystals with unique properties that distinguish them from traditional quantum dots. These nanocrystals have an ABX3 composition, where 'A' can be cesium, methylammonium (MA), or formamidinium (FA); 'B' is typically lead or tin; and 'X' is a halogen ion like chloride, bromide, or iodide. Their remarkable optoelectronic properties, such as high photoluminescence quantum yields and tunable emission across the visible spectrum, make them ideal for applications in light-emitting diodes, lasers, and solar cells."
```

Now we process the text with spaCy:

```python
doc = nlp(perovskite_text)
```

To extract sentences from the processed text we use the *list()* function:

```python
sentences = list(doc.sents)
```


We use *for loop* and *print()* function to output each sentence to show the segmentation:

```python
for sentence in sentences:
   print(sentence.perovskite_text)
```


```
Output: Perovskite nanocrystals are a class of semiconductor nanocrystals with unique properties that distinguish them from traditional quantum dots.
These nanocrystals have an ABX3 composition, where 'A' can be cesium, methylammonium (MA), or formamidinium (FA); 'B' is typically lead or tin; and 'X' is a halogen ion like chloride, bromide, or iodide.
Their remarkable optoelectronic properties, such as high photoluminescence quantum yields and tunable emission across the visible spectrum, make them ideal for applications in light-emitting diodes, lasers, and solar cells.
```



::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Q: Let’s try again by completing the code below to segment sentences from a paragraph about “your field of research”:

```python
import spacy
nlp = _____.load("en_core_web_sm")
# Add the paragraph about your field of research here
text = "___" 
doc = nlp(___)
# Fill in the blank to extract sentences:
sentences = list(______) 
# Fill in the blank to print each sentence
for sentence in sentences:
  print(______)  
```


:::::::::::::::: solution

A:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
# Add the paragraph about your field of research here
text = "***" # varies based on your field of research
doc = nlp(text)
# Fill in the blank to extract sentences:
sentences = list(doc.sents) 
# Fill in the blank to print each sentence
for sentence in sentences:
  print(sentence.text)  
```

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::





::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: Why is text preprocessing necessary for NLP tasks? Think of some examples of NLP tasks that require text preprocessing, such as sentiment analysis, machine translation, or text summarization. How does text preprocessing improve the performance and accuracy of these tasks?

:::::::::::::::::::::::::::::::::::::::::::::::





::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: Use the spaCy library to perform sentence segmentation and tokenization on the following text:

```python
text: "The research (Ref. [1]) focuses on developing perovskite nanocrystals with a bandgap of 1.5 eV, suitable for solar cell applications!". 
```
Print the number of sentences and tokens in the text, and the list of sentences and tokens. You can use the following code to load the *spaCy* library and the English language model:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
```



:::::::::::::::: solution

A: 

```python

import spacy
# Load the English language model:
nlp = spacy.load("en_core_web_sm")

# Define the text with marks, letters, and numbers:
text = "The research (Ref. [1]) focuses on developing perovskite nanocrystals with a bandgap of 1.5 eV, suitable for solar cell applications.!"

# Process the text with spaCy
doc = nlp(text)

# Print the original text:
print("Original text:", text)

# Sentence segmentation:
sentences = list(doc.sents)

# Print the sentences:
print("Sentences:")
for sentence in sentences:
    print(sentence.text)

# Tokenization:
tokens = [token.text for token in doc]

# Print the tokens:
print("Tokens:")
print(tokens)
```

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



## 2.2. Tokeniziation

As already mentioned, in the first episode, Tokenization breaks down text into individual words or tokens, which is a fundamental step for many NLP tasks.




::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: To better understand how it works let’s Match tokens from the provided paragraph about perovskite nanocrystals with similar tokens from another scientific text. This helps in understanding the common vocabulary used in the scientific literature. Using the sentences we listed in the previous section, we can see how Tokenization performs. Assuming 'sentences' is a list of sentences from the previous example, choose a sentence to tokenize:

```python
sentence_to_tokenize = sentences[0]
# Tokenize the chosen sentence by using a list comprehension:
tokens = [token.perovskite_text for token in sentence_to_tokenize]
# We can print the tokens:
print(tokens)
```


Output: ['Perovskite', 'nanocrystals', 'are', 'a', 'class', 'of', 'semiconductor', 'nanocrystals', 'with', 'unique', 'properties', 'that', 'distinguish', 'them', 'from', 'traditional', 'quantum', 'dots', '.']


Tokenization is not just about splitting text into words; it’s about understanding the boundaries of words and symbols in different contexts, which can vary greatly between languages and even within the same language in different settings.


:::::::::::::::::::::::::::::::::::::::::::::::





::: callout

Tokenization is very important for text analysis tasks such as sentiment analysis. Here we can compare two different texts from different fields and see how their associated tokens are different:

```python

perovskite_tokens = [token.text for token in nlp(perovskite_text)]
```

Now, we can add a new text from the trading context for comparison. Tokenization of a trading text can be performed similarly to the previous text.

```python
trading_text = "Trading strategies often involve analyzing patterns and executing trades based on predicted market movements. Successful traders analyze trends and volatility to make informed decisions."

trading_tokens = [token.text for token in nlp(trading_text)]
```

We can see the results by using the *print()* function. The tokens from both texts:

```python
print("Perovskite Tokens:", perovskite_tokens)
print("Trading Tokens:", trading_tokens)
```

```
Output: 
Perovskite Tokens: ['Perovskite', 'nanocrystals', 'are', 'a', 'class', 'of', 'semiconductor', 'nanocrystals', 'with', 'unique', 'properties', 'that', 'distinguish', 'them', 'from', 'traditional', 'quantum', 'dots', '.']
Trading Tokens: ['Trading', 'strategies', 'often', 'involve', 'analyzing', 'patterns', 'and', 'executing', 'trades', 'based', 'on', 'predicted', 'market', 'movements', '.', 'Successful', 'traders', 'analyze', 'trends', 'and', 'volatility', 'to', 'make', 'informed', 'decisions', '.']
```



The tokens from the perovskite text will be specific to materials science, while the trading tokens will include terms related to market analysis. The scientific texts may use more complex and compound words while trading texts might include more action-oriented and analytical language. This comparison helps in understanding the specialized language used in different fields. 

:::





## 2.3. Stemming and Lemmatization

Stemming and lemmatization are techniques used to reduce words to their base or root form, aiding in the normalization of text. As discussed in the previous episode, these two methods are different. Decide whether stemming or lemmatization would be more appropriate for analyzing a set of research texts on perovskite nanocrystals.





::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: From the differences between lemmatization and stemming that we learned in the last episode, which technique will you select to get more accurate text analysis results? Explain why?


:::::::::::::::::::::::::::::::::::::::::::::::



Following our initial example for Tokenization, we can see how lemmatization works. We start with processing the text with *spaCy* to perform lemmatization:

```python

lemmas = [token.lemma_ for token in doc]
```


We can print the original text and the lemmatized text:

```python

print("Original Text:", perovskite_text)
print("Lemmatized Text:", ' '.join(lemmas))
```


Output: 

**Original Text**: Perovskite nanocrystals are a class of semiconductor nanocrystals with unique properties that distinguish them from traditional quantum dots. These nanocrystals have an ABX3 composition, where 'A' can be cesium, methylammonium (MA), or formamidinium (FA); 'B' is typically lead or tin; and 'X' is a halogen ion like chloride, bromide, or iodide. Their remarkable optoelectronic properties, such as high photoluminescence quantum yields and tunable emission across the visible spectrum, make them ideal for applications in light-emitting diodes, lasers, and solar cells.

**Lemmatized Text**: Perovskite nanocrystal be a class of semiconductor nanocrystal with unique property that distinguish they from traditional quantum dot . these nanocrystal have an ABX3 composition , where ' A ' can be cesium , methylammonium ( MA ) , or formamidinium ( FA ) ; ' b ' be typically lead or tin ; and ' x ' be a halogen ion like chloride , bromide , or iodide . their remarkable optoelectronic property , such as high photoluminescence quantum yield and tunable emission across the visible spectrum , make they ideal for application in light - emit diode , laser , and solar cell .





::: callout

The spaCy library does not have stemming capabilities and if we want to compare stemming and lemmatization, we also need to use another language processing library called NLTK (refer to episode 1). 

:::





Based on what we just learned let's compare lemmatization and stemming. First, we need to import the necessary libraries for stemming and lemmatization:

```python

import spacy
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
```

Next, we can create an instance of the PorterStemmer for NLTK and load the English language model for spaCy (similar to what we did earlier in this episode).


```python

stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")
```

We can conduct stemming and lemmatization with identical text data:



:::::::::::::::::::::::::::::::::::::::::: spoiler

### text
text = " Perovskite nanocrystals are a class of semiconductor nanocrystals with unique properties that distinguish them from traditional quantum dots. These nanocrystals have an ABX3 composition, where 'A' can be cesium, methylammonium (MA), or formamidinium (FA); 'B' is typically lead or tin; and 'X' is a halogen ion like chloride, bromide, or iodide. Their remarkable optoelectronic properties, such as high photoluminescence quantum yields and tunable emission across the visible spectrum, make them ideal for applications in light-emitting diodes, lasers, and solar cells."
Before we can stem or lemmatize, we need to tokenize the text.


::::::::::::::::::::::::::::::::::::::::::::::::::





```python

from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)  
# Apply stemming to each token:
stemmed_tokens = [stemmer.stem(token) for token in tokens]
```

For lemmatization, we process the text with spaCy and extract the lemma for each token:


```python

doc = nlp(text)
lemmatized_tokens = [token.lemma_ for token in doc]
```

Finally, we can compare the stemmed and lemmatized tokens:


```python

print("Original Tokens:", tokens)
print("Stemmed Tokens:", stemmed_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)
```



:::::::::::::::::::::::::::::::::::::::::: spoiler

### Output

- Original Tokens: ['Perovskite', 'nanocrystals', 'are', 'a', 'class', 'of', 'semiconductor', 'nanocrystals', 'with', 'unique', 'properties', 'that', 'distinguish', 'them', 'from', 'traditional', 'quantum', 'dots', '.', 'These', 'nanocrystals', 'have', 'an', 'ABX3', 'composition', ',', 'where', "'", 'A', "'", 'can', 'be', 'cesium', ',', 'methylammonium', '(', 'MA', ')', ',', 'or', 'formamidinium', '(', 'FA', ')', ';', "'", 'B', "'", 'is', 'typically', 'lead', 'or', 'tin', ';', 'and', "'", 'X', "'", 'is', 'a', 'halogen', 'ion', 'like', 'chloride', ',', 'bromide', ',', 'or', 'iodide', '.', 'Their', 'remarkable', 'optoelectronic', 'properties', ',', 'such', 'as', 'high', 'photoluminescence', 'quantum', 'yields', 'and', 'tunable', 'emission', 'across', 'the', 'visible', 'spectrum', ',', 'make', 'them', 'ideal', 'for', 'applications', 'in', 'light-emitting', 'diodes', ',', 'lasers', ',', 'and', 'solar', 'cells', '.']

- Stemmed Tokens: ['perovskit', 'nanocryst', 'are', 'a', 'class', 'of', 'semiconductor', 'nanocryst', 'with', 'uniqu', 'properti', 'that', 'distinguish', 'them', 'from', 'tradit', 'quantum', 'dot', '.', 'these', 'nanocryst', 'have', 'an', 'abx3', 'composit', ',', 'where', "'", 'a', "'", 'can', 'be', 'cesium', ',', 'methylammonium', '(', 'ma', ')', ',', 'or', 'formamidinium', '(', 'fa', ')', ';', "'", 'b', "'", 'is', 'typic', 'lead', 'or', 'tin', ';', 'and', "'", 'x', "'", 'is', 'a', 'halogen', 'ion', 'like', 'chlorid', ',', 'bromid', ',', 'or', 'iodid', '.', 'their', 'remark', 'optoelectron', 'properti', ',', 'such', 'as', 'high', 'photoluminesc', 'quantum', 'yield', 'and', 'tunabl', 'emiss', 'across', 'the', 'visibl', 'spectrum', ',', 'make', 'them', 'ideal', 'for', 'applic', 'in', 'light-emit', 'diod', ',', 'laser', ',', 'and', 'solar', 'cell', '.']

- Lemmatized Tokens: ['Perovskite', 'nanocrystal', 'be', 'a', 'class', 'of', 'semiconductor', 'nanocrystal', 'with', 'unique', 'property', 'that', 'distinguish', 'they', 'from', 'traditional', 'quantum', 'dot', '.', 'these', 'nanocrystal', 'have', 'an', 'ABX3', 'composition', ',', 'where', "'", 'A', "'", 'can', 'be', 'cesium', ',', 'methylammonium', '(', 'MA', ')', ',', 'or', 'formamidinium', '(', 'FA', ')', ';', "'", 'b', "'", 'be', 'typically', 'lead', 'or', 'tin', ';', 'and', "'", 'x', "'", 'be', 'a', 'halogen', 'ion', 'like', 'chloride', ',', 'bromide', ',', 'or', 'iodide', '.', 'their', 'remarkable', 'optoelectronic', 'property', ',', 'such', 'as', 'high', 'photoluminescence', 'quantum', 'yield', 'and', 'tunable', 'emission', 'across', 'the', 'visible', 'spectrum', ',', 'make', 'they', 'ideal', 'for', 'application', 'in', 'light', '-', 'emit', 'diode', ',', 'laser', ',', 'and', 'solar', 'cell', '.'] 

::::::::::::::::::::::::::::::::::::::::::::::::::





We can see how stemming often cuts off the end of words, sometimes resulting in non-words, while lemmatization returns the base or dictionary form of the word. For example, stemming might reduce **“properties”** to **“properti”** while lemmatization would correctly identify the lemma as **“property”**. Lemmatization provides a more readable and meaningful result, which is particularly useful in NLP tasks that require understanding the context and meaning of words.





::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: Use the spaCy library to perform lemmatization on the following text: "Perovskite nanocrystals are a promising class of materials for optoelectronic applications due to their tunable bandgaps and high photoluminescence efficiencies." Print the original text and the lemmatized text. You can use the following code to load the spacy library and the English language model:

```python

import spacy
nlp = spacy.load("en_core_web_sm")
```




:::::::::::::::: solution

A: 

```python

import spacy
# Load the English language model:
nlp = spacy.load("en_core_web_sm")

# Define the text:
text = "Perovskite nanocrystals are a promising class of materials for optoelectronic applications due to their tunable bandgaps and high photoluminescence efficiencies."

# Process the text with spaCy:
doc = nlp(text)

# Print the original text:
print("Original text:", text)
# Print the lemmatized text:
lemmatized_text = " ".join([token.lemma_ for token in doc])
print("Lemmatized text:", lemmatized_text)

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::





## 2.4. Stop-words Removal

Removing stop-words, which are common words that add little value to the analysis (such as ‘and’ and ‘the’), helps focus on the important content. Assuming 'doc' is the processed text from the previous example for ‘perovskite nanocrystals’, we can define a list to hold non-stop words list comprehensions:


```python

filtered_sentence = [word for word in doc if not word.is_stop]
# print the filtered sentence and see how it is changed:
print("Filtered sentence:", filtered_sentence)
```

Output: 
Filtered sentence: [Perovskite, nanocrystals, class, semiconductor, nanocrystals, unique, properties, distinguish, traditional, quantum, dots, ., nanocrystals, ABX3, composition, ,, ', ', cesium, ,, methylammonium, (, MA, ), ,, formamidinium, (, FA, ), ;, ', B, ', typically, lead, tin, ;, ', X, ', halogen, ion, like, chloride, ,, bromide, ,, iodide, ., remarkable, optoelectronic, properties, ,, high, photoluminescence, quantum, yields, tunable, emission, visible, spectrum, ,, ideal, applications, light, -, emitting, diodes, ,, lasers, ,, solar, cells, .]


List comprehensions provide a convenient method for rapidly generating lists based on a straightforward condition. 





::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Q: To see how list comprehensions are created, fill in the missing parts of the code to remove stop-words from a given sentence.

```python
import _____
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a very simple and short sentence.")
filtered_sentence = [____ for ____ in doc if not ____]
print("Filtered sentence:", filtered_sentence)
```



:::::::::::::::: solution

A: 

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a very simple and short sentence.")
filtered_sentence = [word for words in doc if not word.is_stop]
print("Filtered sentence:", filtered_sentence)
```

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::




::: callout

While stopwords are often removed to improve analysis, they can be important for certain tasks like sentiment analysis, where the word ‘not’ can change the entire meaning of a sentence.

:::





::: callout

it is important to note that tokenization is just the beginning. In modern NLP, vectorization, and embeddings play a pivotal role in capturing the context and meaning of text.

Vectorization is the process of converting tokens into a numerical format that machine learning models can understand. This often involves creating a bag-of-words model, where each token is represented by a unique number in a vector. Embeddings are advanced representations where words are mapped to vectors of real numbers. They capture not just the presence of tokens but also the semantic relationships between them. This is achieved through techniques like Word2Vec, GloVe, or BERT, which we will explore in the second part of our workshop.

These embeddings allow models to understand the text in a more nuanced way, leading to better performance on tasks such as sentiment analysis, machine translation, and more.

*Stay tuned for our next session, where we will dive deeper into how we can use vectorization and embeddings to enhance our NLP models and truly capture the richness of language.*

:::





::::::::::::::::::::::::::::::::::::: keypoints 

- Text preprocessing is essential for cleaning and standardizing text data.
- Techniques like sentence segmentation, tokenization, stemming, and lemmatization are fundamental to text preprocessing.
- Removing stop-words helps in focusing on the important words in text analysis.
- Tokenization splits sentences into tokens, which are the basic units for further processing.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
