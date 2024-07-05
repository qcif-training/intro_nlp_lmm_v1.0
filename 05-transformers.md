---
title: 'Transformers for Natural Language Processing'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do Transformers work?
- How can I use Transformers for text analysis?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- To be able to describe Transformers’ architecture.
- To be able to implement sentiment analysis, and text summarization using transformers.

::::::::::::::::::::::::::::::::::::::::::::::::

Transformers have revolutionized the field of NLP since their introduction by the Google team in 2017. Unlike previous models that processed text sequentially, Transformers use an attention mechanism to process all words at once, allowing them to capture context more effectively. This parallel processing capability enables Transformers to handle long-range dependencies and understand the nuances of language better than their predecessors. For now, try to recognize the building blocks of the general structure of a transformer


![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/7ec8bb01-aa7e-4378-af45-ae28fbe8916b)



## 5.1. Introduction to Artificial Neural Networks

To understand how Transformers work we also need to learn about artificial neural networks (ANNs). Imagine a neural network as a team of workers in a factory. Each worker (neuron) has a specific task (processing information), and they pass their work along to the next person in line until the final product (output) is created. 

Just like a well-organized assembly line, a neural network processes information in stages, with each neuron contributing to the final result. 



::::::::::::::::::::::::::::::::::::: challenge

### Activity

Teamwork: Take a look at the architecture of a simple ANN below. Identify the underlying layers and components of this ANN and add the correct name label to each one.

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/445b963e-caf1-451d-8edc-e686f8950ae5)


:::::::::::::::::::::::: solution 

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/523924b5-055b-4125-8bce-aa2be2b38ca8)


:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



:::::::::::::::::::::::::::::::::::::::::: spoiler
### What is Multilayer Perceptron then?

In the context of machine learning, a multilayer perceptron (MLP) is indeed a fully connected multi-layer neural network and is a classic example of a feedforward artificial neural network (ANN). It typically includes an input layer, one or more hidden layers, and an output layer. When an MLP has more than one hidden layer, it can be considered a deep ANN, part of a broader category known as deep learning. 

::::::::::::::::::::::::::::::::::::::::::::::::::



::: callout

### Summation and Activation Function
If we zoom into a neuron in the hidden layer, we can see the mathematical operations (weights summation and activation function). An input is transformed at each hidden layer node through a process that multiplies the input (x_i) by learned weights (w_i), adds a bias (b), and then applies an activation function to determine the node’s output. This output is either passed on to the next layer or contributes to the final output of the network. Essentially, each node performs a small calculation that, when combined with the operations of other nodes, allows the network to process complex patterns and data. 

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/d1006506-ac54-43bc-b6e9-5181ca98be36)

::: 



:::::::::::::::::::::::::::::::::::::::::: spoiler

### What happens next? How to optimize an ANN? 

Backpropagation is an algorithmic cornerstone in the training of ANNs, serving as a method for optimizing weights and biases through gradient descent. Conceptually, it is akin to an iterative refinement process where the network’s output error is propagated backward, layer by layer, using the chain rule of calculus. This backward flow of error information allows for the computation of gradients, which inform the magnitude and direction of adjustments to be made to the network’s parameters. The objective is to iteratively reduce the differences between the predicted output and the actual target values. This systematic adjustment of parameters, guided by error gradients, incrementally leads to a more accurate ANN model.

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/5ae4d845-c3d2-42df-87f0-e928be9ba64b)


::::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Teamwork: When we talk about ANNs, we also consider their parameters. But what are the parameters? Draw a small neural network with 3 following layers: x1

- Input Layer: 3 neurons
- Hidden Layer: 4 neurons
- Output Layer: 1 neurons

1. Connect each neuron in the input layer to every neuron in the hidden layer (next layer). How many connections (weights) do we have?
2. Now, add a bias for each neuron in the hidden layer. How many biases do we have?
3. Repeat the process for the hidden layer to the output layer.

:::::::::::::::::::::::: solution 

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/144a8f7d-c1ac-4b57-8688-b5280cabd59c)


- (3 { neurons} x 4 { neurons} + 4{ biases}) = 16 
- (4 { neurons} x 1 { neurons} + 1{ biases}) = 5
- Total parameters for this network: (16 + 5 = 21)

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: challenge

### Challenge


Q: Add another hidden layer with 4 neurons to the previous ANN and calculate the number of parameters.

:::::::::::::::::::::::: solution 

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/39820eb0-7959-4ed1-bdfc-69131ab1a834)


We would add:
- (4 * 4) weights from the first to the second hidden layer
- (4) biases for the new hidden layer
- (4 * 1) weights from the second hidden layer to the output layer (we already counted the biases for the output layer)

That’s an additional (16 + 4 = 20) parameters, bringing our total to (21 + 20 = 41) parameters.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



## 5.2. Transformers

As mentioned in the introduction, Most of the recent NLP models are built based on Transformers. Building on our understanding of ANNs, let’s explore the architecture of transformers. Transformers consist of several key components that work together to process and generate data.


::::::::::::::::::::::::::::::::::::: challenge
### Activity

Teamwork: We go back to the first figure of this episode. In the simplified schematic below, write the function of each component in the allocated textbox:

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/e94c677a-53c5-4da3-87ac-e45960429986)


:::::::::::::::::::::::: solution 

A:
![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/25823a7c-6059-4c83-b592-a273fa59b9ec)

Briefly, we can say:

- Encoder: Processes input text into contextualized representations, enabling the understanding of the context within the input sequence. It is like the ‘listener’ in a conversation, taking in information and understanding it.
- Decoder: Generates output sequences by translating the contextualized representations from the encoder into coherent text, often using mechanisms like masked multi-head attention and encoder-decoder attention to maintain sequence order and coherence. This acts as the ‘speaker’ in the conversation, generating the output based on the information processed by the encoder.
- Positional Encoding: Adds unique information to each word embedding, indicating the word’s position in the sequence, which is essential for the model to maintain the order of words and understand their relative positions within a sentence
- Input Embedding: The input text is converted into vectors that the model can understand. Think of it as translating words into a secret code that the transformer can read.
- Output Embedding: Similar to input embedding, but for the output text. It translates the transformer’s secret code back into words we can understand.
- Softmax Output: Applies the softmax function to the final layer’s outputs to convert them into a probability distribution, which helps in tasks like classification and sequence generation by selecting the most likely next word or class. It is like choosing the best response in a conversation from many options.


:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



::: callout

### Attention Mechanism

So far, we have learned what the architecture of a transformer block looks like. However, for simplicity, many parts of this architecture have not been considered. 

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/559580aa-f2c9-4ec0-b87d-7e1438839431)

In the following section, we will show the underlying components of a transformer.

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/2b325dc1-20d8-4bac-91b8-030067ee8097)

For more details see [source](https://arxiv.org/abs/1706.03762).

Attention mechanisms in transformers, allow LLMs to focus on different parts of the input text to understand context and relationships between words. The concept of ‘attention’ in encoders and decoders is akin to the selective focus of ‘fast reading,’ where one zeroes in on crucial information and disregards the irrelevant. This mechanism adapts to the context of a query, emphasizing different words or tokens based on the query’s intent. For instance, in the sentence “Sarah went to a restaurant to meet her friend that night,” the words highlighted would vary depending on whether the question is about the action (What?), location (Where?), individuals involved (Who?), or time (When?).


![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/f28a1a04-40f7-4ad3-a846-4f0d4c4ddee4)
[source](https://medium.com/@hunter-j-phillips/multi-head-attention-7924371d477a)


In transformer models, this selective focus is achieved through ‘queries,’ ‘keys,’ and ‘values,’ all represented as vectors. A query vector seeks out the closest key vectors, which are encoded representations of values. The relationship between words, like ‘where’ and ‘restaurant,’ is determined by their frequency of co-occurrence in sentences, allowing the model to assign greater attention to ‘restaurant’ when the query pertains to a location. This dynamic adjustment of focus enables transformers to process language with a nuanced understanding of context and relevance.

:::



::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: Have you heard of any other applications of the Transformers rather than in NLPs? Explain why transformers can be useful for other AI applications. Share your thoughts and findings with other groups.


:::::::::::::::: solution

A: Transformers, initially popular in NLP, have found applications beyond text analysis. They excel in computer vision, speech recognition, and even genomics. Their versatility extends to music generation and recommendation systems. Transformers’ innovative architecture allows them to adapt to diverse tasks, revolutionizing AI applications.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



::: callout

### Transformers in Text Translation
Imagine you want to translate the sentence “What time is it?” from English to German using a transformer.
The input embedding layer converts each English word into a vector.
The six layers of encoders process these vectors, understanding the context of the sentence.
The six layers of decoders then start generating the German translation, one word at a time.

![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/09e7ae0f-bfc1-4d7c-be5f-443b97c05bd3)

For each word, the Softmax output predicts the most likely next word in German.
The output embedding layer converts these predictions back into readable German words.
By the end, you get the German translation of **“What time is it?”** as **“Wie spät ist es?”**

:::


::: callout 

### Pipelines
The pipeline module from Hugging Face’s [transformers library](https://huggingface.co/docs/transformers/en/main_classes/pipelines) is a high-level API that simplifies the use of complex machine learning models for a variety of NLP tasks. It is a versatile tool for NLP tasks, enabling users to perform **text generation**, **sentiment analysis**, **question answering**, **summarization**, and **translation** with minimal code. By abstracting away the intricacies of model selection, tokenization, and output generation, the pipeline module makes state-of-the-art AI accessible to developers of all skill levels, allowing them to harness the power of language models efficiently and intuitively.
:::


:::::::::::::::::::::::::::::::::::::::::: spoiler

### What are other sequential learning models? 

Transformers are essential for NLP tasks because they overcome the limitations of earlier models like recurrent neural networks (RNNs) and long short-term memory models (LSTMs), which struggled with long sequences and were computationally intensive respectively. Transformers, in contrast to the sequential input processing of RNNs, handle entire sequences simultaneously. This parallel processing capability enables data scientists to employ GPUs to train large language models (LLMs) based on transformers, which markedly decreases the duration of training.

![rnn-transf-nlp](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/a0a429e4-c87e-4529-9151-781dd566c800)
[source](https://thegradient.pub/transformers-are-graph-neural-networks/)

::::::::::::::::::::::::::::::::::::::::::::::::::



## 5.3. Sentiment Analysis

Sentiment analysis is a powerful tool in NLP that helps determine the emotional tone behind the text. It is used to understand opinions, sentiments, emotions, and attitudes from various entities and classify them according to their polarity. 



::::::::::::::::::::::::::::::::::::: challenge
### Activity

Teamwork: How do you categorize the following text in terms of positive and negative sounding? Select an Emoji.

*“A research team has unveiled a novel ligand exchange technique that enables the synthesis of organic cation-based perovskite quantum dots (PQDs), ensuring exceptional stability while suppressing internal defects in the photoactive layer of solar cells.”* [source](https://www.sciencedaily.com/releases/2024/02/240221160400.htm)


![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/897d0938-d84f-4189-afac-b8f244e16b46)


::::::::::::::::::::::::::::::::::::::::::::::::



Computer models can do this job for us! Let’s see how it works through a step-by-step example: First, install the required libraries and pipelines:


```python

pip install transformers
from transformers import pipeline
```

Now, initialize the sentiment analysis pipeline and analyze the sentiment of a sample text:

```python

sentiment_pipeline = pipeline('sentiment-analysis')
text = " A research team has unveiled a novel ligand exchange technique that enables the synthesis of organic cation-based perovskite quantum dots (PQDs), ensuring exceptional stability while suppressing internal defects in the photoactive layer of solar cells."
sentiment = sentiment_pipeline(text)
```

After the analysis is completed, you can print out the results:

```python

print(f"Sentiment: {sentiment[0]['label']}, Confidence: {sentiment[0]['score']:.2f}")

# Output

Output: Sentiment: POSITIVE, Confidence: 1.00

```

In this example, the sentiment analysis pipeline from the Hugging Face library is used to analyze the sentiment of a research paper abstract. The model predicts the sentiment as positive, negative, or neutral, along with a confidence score. This can be particularly useful for gauging the reception of research papers in a field.



::::::::::::::::::::::::::::::::::::: challenge
### Activity

Teamwork: Fill in the blanks to complete the sentiment analysis process:
Install the __________ library for sentiment analysis.
Use the __________ function to create a sentiment analysis pipeline.
The sentiment analysis model will output a __________ and a __________ score.


::::::::::::::::::::::::::::::::::::::::::::::::



::: callout

### VADRER

Valence Aware Dictionary and sEntiment Reasoner (VADER) is a lexicon and rule-based sentiment analysis tool that is particularly attuned to sentiments expressed in social media. VADER analyzes the sentiment of the text and returns a dictionary with scores for negative, neutral, positive, and a compound score that aggregates them. It is useful for quick sentiment analysis, especially on social media texts. Let’s how we can use this framework. 

First, we need to import the SentimentIntensityAnalyzer module from VADER library:

```python

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Initialize VADER sentiment intensity analyzer:
analyzer = SentimentIntensityAnalyzer()

# We use the same sample text:
text = " A research team has unveiled a novel ligand exchange technique that enables the synthesis of organic cation-based perovskite quantum dots (PQDs), ensuring exceptional stability while suppressing internal defects in the photoactive layer of solar cells."

# Now we can analyze sentiment:
vader_sentiment = analyzer.polarity_scores(text)

# Print the sentiment:
print(f"Sentiment: {vader_sentiment}")
Output: Sentiment: {'neg': 0.069, 'neu': 0.818, 'pos': 0.113, 'compound': 0.1779}
```

:::



::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: Which framework do you think could be more helpful for research applications? Elaborate your opinion. Share your thoughts with other team members.

:::::::::::::::: solution

A: Transformers use deep learning models that can understand context and nuances of language, making them suitable for complex and lengthy texts. They can be particularly useful for sentiment analysis of research papers, as they can understand the complex language and context often found in academic writing. This allows for a more nuanced understanding of the sentiment conveyed in the papers. VADER, on the other hand, is a rule-based model that excels in analyzing short texts with clear sentiment expressions, often found in social media.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Use the transformers library to perform sentiment analysis on the following text: 

*"Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers."*


Print the original text and the sentiment score and label. You can use the following code to load the transformers library and the pre-trained model and tokenizer for sentiment analysis:

```python

from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis")

```


:::::::::::::::: solution

A: 

```python
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis")
text = "This book is amazing. It is well-written, engaging, and informative. I learned a lot from reading it and I highly recommend it to anyone interested in natural language processing."
print(text)
print(sentiment_analysis(text))
```
Output:

```python

output: "Perovskite nanocrystals have emerged as a promising class of materials for next-generation optoelectronic devices due to their unique properties. Their crystal structure allows for tunable bandgaps, which are the energy differences between occupied and unoccupied electronic states. This tunability enables the creation of materials that can absorb and emit light across a wide range of the electromagnetic spectrum, making them suitable for applications like solar cells, light-emitting diodes (LEDs), and lasers."

[{'label': 'POSITIVE', 'score': 0.9998656511306763}]
```


:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::





::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Comparing Transformer with VADER on a large size text. Use the Huggingface library database.


:::::::::::::::: solution

A: 

```python
from transformers import pipeline

```


:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



## 5.4. Text Summarization


Text summarization is the process of distilling the most important information from a source (or sources) to produce an abbreviated version for a particular user and task. It can be broadly classified into two types: extractive and abstractive summarization.



::::::::::::::::::::::::::::::::::::: challenge

### Discussion

How extractive and abstractive summarization methods are different? Connect the following text boxes to the correct category. Share your results with other group members.


![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/1630ffcf-90c8-49df-9abe-98a739fd58ef)


:::::::::::::::: solution

A: 
![image](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/166c2025-14b0-4f1b-aee5-3f46d4f3c8b4)



:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



Now, let’s see how to use the Hugging Face Transformers library to perform abstractive summarization.
First, from the transformers import pipeline:

```python

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

```

Input a sample text from an article from [source](https://www.sciencedaily.com/releases/2024/02/240221160400.htm#:~:text=Summary%3A,photoactive%20layer%20of%20solar%20cells.):

:::::::::::::::::::::::::::::::::::::::::: spoiler

### Input Text


text = "A groundbreaking research breakthrough in solar energy has propelled the development of the world's most efficient quantum dot (QD) solar cell, marking a significant leap towards the commercialization of next-generation solar cells. This cutting-edge QD solution and device have demonstrated exceptional performance, retaining their efficiency even after long-term storage. Led by Professor Sung-Yeon Jang from the School of Energy and Chemical Engineering at UNIST, a team of researchers has unveiled a novel ligand exchange technique. This innovative approach enables the synthesis of organic cation-based perovskite quantum dots (PQDs), ensuring exceptional stability while suppressing internal defects in the photoactive layer of solar cells. Our developed technology has achieved an impressive 18.1% efficiency in QD solar cells," stated Professor Jang. This remarkable achievement represents the highest efficiency among quantum dot solar cells recognized by the National Renewable Energy Laboratory (NREL) in the United States. The increasing interest in related fields is evident, as last year, three scientists who discovered and developed QDs, as advanced nanotechnology products, were awarded the Nobel Prize in Chemistry. QDs are semiconducting nanocrystals with typical dimensions ranging from several to tens of nanometers, capable of controlling photoelectric properties based on their particle size. PQDs, in particular, have garnered significant attention from researchers due to their outstanding photoelectric properties. Furthermore, their manufacturing process involves simple spraying or application to a solvent, eliminating the need for the growth process on substrates. This streamlined approach allows for high-quality production in various manufacturing environments. However, the practical use of QDs as solar cells necessitates a technology that reduces the distance between QDs through ligand exchange, a process that binds a large molecule, such as a ligand receptor, to the surface of a QD. Organic PQDs face notable challenges, including defects in their crystals and surfaces during the substitution process. As a result, inorganic PQDs with limited efficiency of up to 16% have been predominantly utilized as materials for solar cells. In this study, the research team employed an alkyl ammonium iodide-based ligand exchange strategy, effectively substituting ligands for organic PQDs with excellent solar utilization. This breakthrough enables the creation of a photoactive layer of QDs for solar cells with high substitution efficiency and controlled defects. Consequently, the efficiency of organic PQDs, previously limited to 13% using existing ligand substitution technology, has been significantly improved to 18.1%. Moreover, these solar cells demonstrate exceptional stability, maintaining their performance even after long-term storage for over two years. The newly-developed organic PQD solar cells exhibit both high efficiency and stability simultaneously. Previous research on QD solar cells predominantly employed inorganic PQDs," remarked Sang-Hak Lee, the first author of the study. Through this study, we have demonstrated the potential by addressing the challenges associated with organic PQDs, which have proven difficult to utilize. This study presents a new direction for the ligand exchange method in organic PQDs, serving as a catalyst to revolutionize the field of QD solar cell material research in the future," commented Professor Jang. The findings of this study, co-authored by Dr. Javid Aqoma Khoiruddin and Sang-Hak Lee, have been published online in Nature Energy on January 27, 2024. The research was made possible through the support of the 'Basic Research Laboratory (BRL)' and 'Mid-Career Researcher Program,' as well as the 'Nano·Material Technology Development Program,' funded by the National Research Foundation of Korea (NRF) under the Ministry of Science and ICT (MSIT). It has also received support through the 'Global Basic Research Lab Project."


::::::::::::::::::::::::::::::::::::::::::::::::::


Now we can perform summarization and print the results:


```python

summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
# Print the summary:
print("Summary:", summary[0]['summary_text'])


Output: 


```



::: callout

### Sumy for summarization

Sumy is a Python library for extractive summarization. It uses algorithms like LSA to rank sentences based on their importance and creates a summary by selecting the top-ranked sentences. We can see how it works in practice:
We start with importing the PlaintextParser and LsaSummarizer modules:

```python

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

```

To create a parser we use the same text sample from an article from [source](https://www.sciencedaily.com/releases/2024/02/240221160400.htm#:~:text=Summary%3A,photoactive%20layer%20of%20solar%20cells.):


```python

parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Next, we initialize the LSA summarize:

summarizer = LsaSummarizer()

# Summarize the text and print the results

summary = summarizer(parser.document, 5)

for sentence in summary:
    print(sentence)

Output:

```

Sumy extracts key sentences from the original text, which can be quicker but may lack the cohesiveness of an abstractive summary. On the other hand, Transformer is suitable for generating a new summary that captures the text’s essence in a coherent and often more readable form.


:::



::::::::::::::::::::::::::::::::::::: challenge

### Activity 

Teamwork: Which framework could be more useful for text summarizations in your field of research? Explain why?



:::::::::::::::::::::::: solution 

A: Transformers are particularly useful for summarizing research papers and documents where understanding the context and generating a coherent summary is crucial. They can produce summaries that are not only concise but also maintain the narrative flow, making them more readable. Sumy, while quicker and less resource-intensive, is best suited for scenarios where extracting key information without the need for narrative flow is acceptable.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::




::::::::::::::::::::::::::::::::::::: challenge

### Challenge

Use the transformers library to perform text summarization on the following text [generated by Copilot]: 

:::::::::::::::::::::::::::::::::::::::::: spoiler

### Input Text

text: "Perovskite nanocrystals are a class of semiconductor nanocrystals that have attracted a lot of attention in recent years due to their unique optical and electronic properties. Perovskite nanocrystals have an ABX3 composition, where A is a monovalent cation (such as cesium, methylammonium, or formamidinium), B is a divalent metal (such as lead or tin), and X is a halide (such as chloride, bromide, or iodide). Perovskite nanocrystals can emit brightly across the entire visible spectrum, with tunable colors depending on their composition and size. They also have high quantum yields, fast radiative decay rates, and narrow emission line widths, making them ideal candidates for various optoelectronic applications.
The first report of perovskite nanocrystals was published in 2014 by Protesescu et al., who synthesized cesium lead halide nanocrystals using a hot-injection method. They demonstrated that the nanocrystals had cubic or orthorhombic crystal structures, depending on the halide ratio, and that they exhibited strong photoluminescence with quantum yields up to 90%. They also showed that the emission wavelength could be tuned from 410 nm to 700 nm by changing the halide composition or the nanocrystal size. Since then, many other groups have developed various synthetic methods and strategies to control the shape, size, composition, and surface chemistry of perovskite nanocrystals. One of the remarkable features of perovskite nanocrystals is their defect tolerance, which means that they can maintain high luminescence even with a high density of surface or bulk defects. This is in contrast to other semiconductor nanocrystals, such as CdSe, which require surface passivation to prevent non-radiative recombination and quenching of the emission. The defect tolerance of perovskite nanocrystals is attributed to their electronic band structure, which has a large density of states near the band edges and a small effective mass of the charge carriers. These factors reduce the formation energy and the localization of defects and enhance the radiative recombination rate of the excitons. Another interesting aspect of perovskite nanocrystals is their weak quantum confinement, which means that their emission properties are not strongly affected by their size. This is because the exciton binding energy of perovskite nanocrystals is much larger than the quantum confinement energy, and thus the excitons are localized within a few unit cells regardless of the nanocrystal size. As a result, perovskite nanocrystals can exhibit narrow emission line widths even with a large size distribution, which simplifies the synthesis and purification processes. Moreover, perovskite nanocrystals can show dual emission from both the band edge and the surface states, which can be exploited for color tuning and white light generation. Perovskite nanocrystals have been applied to a wide range of photonic devices, such as light-emitting diodes, lasers, solar cells, photodetectors, and scintillators. Perovskite nanocrystals can offer high brightness, color purity, and stability as light emitters, and can be integrated with various substrates and architectures. Perovskite nanocrystals can also act as efficient light absorbers and charge transporters and can be coupled with other materials to enhance the performance and functionality of the devices. Perovskite nanocrystals have shown promising results in terms of efficiency, stability, and versatility in these applications. However, perovskite nanocrystals also face some challenges and limitations, such as the toxicity of lead, the instability under ambient conditions, the hysteresis and degradation under electrical or optical stress, and the reproducibility and scalability of the synthesis and fabrication methods. These issues need to be addressed and overcome to realize the full potential of perovskite nanocrystals in practical devices. Therefore, further research and development are needed to improve the material quality, stability, and compatibility of perovskite nanocrystals, and to explore new compositions, structures, and functionalities of these fascinating nanomaterials."

::::::::::::::::::::::::::::::::::::::::::::::::::


Print the summarized text. 

```python

from transformers import pipeline
summarizer = pipeline("summarization")
...


```


:::::::::::::::::::::::: solution 

A: You can use the following code to load the transformers library and the pre-trained model and tokenizer for text summarization:


```python

from transformers import pipeline
summarizer = pipeline("summarization")
text = " Perovskite nanocrystals are a class of semiconductor nanocrystals that have attracted…

Output:


```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: keypoints 


- Transformers revolutionized NLP by processing words in parallel through an attention mechanism, capturing context more effectively than sequential models
- The summation and activation function within a neuron transform inputs through weighted sums and biases, followed by an activation function to produce an output.
- Transformers consist of encoders, decoders, positional encoding, input/output embedding, and softmax output, working together to process and generate data.
- Transformers are not limited to NLP and can be applied to other AI applications due to their ability to handle complex data patterns.
- Sentiment analysis and text summarization are practical applications of transformers in NLP, enabling the analysis of emotional tone and the creation of concise summaries from large texts.


::::::::::::::::::::::::::::::::::::::::::::::::



<!-- Collect your link references at the bottom of your document -->

