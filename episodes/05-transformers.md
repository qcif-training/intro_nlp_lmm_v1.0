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

### Activity

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


:::::::::::::::::::::::::::::::::::::::::: spoiler

### What are other sequential learning models? 

Transformers are essential for NLP tasks because they overcome the limitations of earlier models like recurrent neural networks (RNNs) and long short-term memory models (LSTMs), which struggled with long sequences and were computationally intensive respectively. Transformers, in contrast to the sequential input processing of RNNs, handle entire sequences simultaneously. This parallel processing capability enables data scientists to employ GPUs to train large language models (LLMs) based on transformers, which markedly decreases the duration of training.

![rnn-transf-nlp](https://github.com/qcif-training/intro_nlp_lmm_v1.0/assets/45458783/a0a429e4-c87e-4529-9151-781dd566c800)
[source](https://thegradient.pub/transformers-are-graph-neural-networks/)

::::::::::::::::::::::::::::::::::::::::::::::::::


## 5.3. Semantic Analysis

Sentiment analysis is a powerful tool in NLP that helps determine the emotional tone behind the text. It is used to understand opinions, sentiments, emotions, and attitudes from various entities and classify them according to their polarity. 


::::::::::::::::::::::::::::::::::::: activity
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

Output: Output: Sentiment: POSITIVE, Confidence: 1.00

```

In this example, the sentiment analysis pipeline from the Hugging Face library is used to analyze the sentiment of a research paper abstract. The model predicts the sentiment as positive, negative, or neutral, along with a confidence score. This can be particularly useful for gauging the reception of research papers in a field.
















#### 1. Prompt Optimization:
To elicit specific and accurate responses from LLMs by designing prompts strategically. 

*Zero-shot Prompting*: This is the simplest form of prompting where the LLM is given a task or question without any context or examples. It relies on the LLM’s pre-existing knowledge to generate a response. 

Example: “What is the capital of France?” The LLM would respond with “Paris” based on its internal knowledge. 

*Few-shot Prompting*: In this technique, the LLM is provided with a few examples to demonstrate the expected response format or content. 

Example: To determine sentiment, you might provide examples like “I love sunny days. (+1)” and “I hate traffic. (-1)” before asking the LLM to analyze a new sentence.

#### 2. Retrieval Augmented Generation (RAG):
To supplement the LLM’s generative capabilities with information retrieved from external databases or documents. 

*Retrieval*: The LLM queries a database to find relevant information that can inform its response. 

Example: If asked about recent scientific discoveries, the LLM might retrieve articles or papers on the topic. 

*Generation*: After retrieving the information, the LLM integrates it into a coherent response. 

Example: Using the retrieved scientific articles, the LLM could generate a summary of the latest findings in a particular field.

#### 3. Fine-Tuning: 
To adapt a general-purpose LLM to excel at a specific task or within a particular domain. 

*Language Modeling Task Fine-Tuning*: This involves training the LLM on a large corpus of text to improve its ability to predict the next word or phrase in a sentence. 

Example: An LLM fine-tuned on legal documents would become better at generating text that resembles legal writing. 

*Supervised Q&A Fine-Tuning*: Here, the LLM is trained on a dataset of question-answer pairs to enhance its performance on Q&A tasks. 

Example: An LLM fine-tuned with medical Q&A pairs would provide more accurate responses to health-related inquiries.

#### 4.	Training from Scratch: 
Builds a model specifically for a domain, using relevant data from the ground up.

::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Which approach do you think is more computation-intensive? Which is more accurate? How are these qualities related?  Evaluate the trade-offs between fine-tuning and other approaches.

![](fig/transformers_3.png)

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Discussion

What is DSL and why are they useful for research tasks? Think of some examples of NLP tasks that require domain-specific LLMs, such as literature review, patent analysis, or material discovery. How do domain-specific LLMs improve the performance and accuracy of these tasks?

![](fig/transformers_4.png)


::::::::::::::::::::::::::::::::::::::::::::::::

## 7.2. Prompting

For research applications where highly reliable answers are crucial, Prompt Engineering combined with Retrieval-Augmented Generation (RAG) is often the most suitable approach. This combination allows for flexibility and high-quality outputs by leveraging both the generative capabilities of LLMs and the precision of domain-specific data sources:

```python
Install the Hugging Face libraries
!pip install transformers datasets

from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Example research question
question = "What is the role of CRISPR-Cas9 in genome editing?"

# Candidate topics to classify the question
topics = ["Biology", "Technology", "Healthcare", "Genetics", "Ethics"]

# Perform zero-shot classification
result = classifier(question, candidate_labels=topics)

# Output the results
print(f"Question: {question}")
print("Classified under topics with the following scores:")
for label, score in zip(result['labels'], result['scores']):
print(f"{label}: {score:.4f}")

```

::::::::::::::::::::::::::::::::::::::::: spoiler 

## Heads-up: Be careful when fine-tuning a model

When fine-tuning a BERT model from Hugging Face, for instance, it is essential to approach the process with precision and care. 

Begin by thoroughly understanding **BERT’s architecture** and the specific task at hand to select the most suitable model variant and hyperparameters. 

**Prepare your dataset** meticulously, ensuring it is clean, well-represented, and split correctly to avoid **data leakage and overfitting**. 

Hyperparameter selection, such as learning rates and batch sizes, should be made with consideration, and **regularization** techniques like dropout should be employed to enhance the model’s ability to generalize. 

**Evaluate** the model’s performance using appropriate metrics and address any class imbalances with weighted loss functions or similar strategies. 

Save checkpoints to preserve progress and document every step of the fine-tuning process for transparency and reproducibility. 

**Ethical considerations** are paramount; strive for a model that is fair and unbiased. 

Ensure compliance with data protection regulations, especially when handling sensitive information. 

Lastly, manage **computational resources** wisely and engage with the Hugging Face community for additional support. Fine-tuning is iterative, and success often comes through continuous experimentation and learning.

::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Guess the following architecture belongs to which optimization strategy:

![](fig/transformers_5.png)

Figure. LLMs optimization (source)

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Discussion

What are the challenges and trade-offs of domain-specific LLMs, such as data availability, model size, and complexity? Consider some of the factors that affect the quality and reliability of domain-specific LLMs, such as the amount and quality of domain-specific data, the computational resources and time required for training or fine-tuning, and the generalization and robustness of the model. How do these factors pose problems or difficulties for domain-specific LLMs and how can we overcome them?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Discussion

Teamwork: What are some available approaches for creating domain-specific LLMs, such as fine-tuning and knowledge distillation? Consider some of the main steps and techniques for creating domain-specific LLMs, such as selecting a general LLM, collecting and preparing domain-specific data, training or fine-tuning the model, and evaluating and deploying the model. How do these approaches differ from each other and what are their advantages and disadvantages?

::::::::::::::::::::::::::::::::::::::::::::::::

### Example:
Now let’s try One-shot and Few-shot prompting examples and see how it can help us to enhance the sensitivity of the LLM to our field of study: One-shot prompting involves providing the model with a single example to follow. It’s like giving the model a hint about what you expect. We will go through an example using Hugging Face’s transformers library:

```python
from transformers import pipeline

# Load a pre-trained model and tokenizer
model_name = "gpt2"
generator = pipeline('text-generation', model=model_name)

# One-shot example
prompt = "Translate 'Hello, how are you?' to French:\nBonjour, comment ça va?\nTranslate 'I am learning new things every day' to French:"
result = generator(prompt, max_length=100)

# Output the result
print(result[0]['generated_text'])
```

In this example, we provide the model with one translation example and then ask it to translate a new sentence. The model uses the context from the one-shot example to generate the translation. But what if we have a Few-Shot Prompting? Few-shot prompting gives the model several examples to learn from. This can improve the model’s ability to understand and complete the task. Here is how you can implement few-shot prompting:

```python
from transformers import pipeline

# Load a pre-trained model and tokenizer
model_name = "gpt2"
generator = pipeline('text-generation', model=model_name)

# Few-shot examples
prompt = """\
Q: What is the capital of France?
A: Paris.

Q: What is the largest mammal?
A: Blue whale.

Q: What is the human body's largest organ?
A: The skin.

Q: What is the currency of Japan?
A:"""
result = generator(prompt, max_length=100)

# Output the result
print(result[0]['generated_text'])
```

In this few-shot example, we provide the model with three question-answer pairs before posing a new question. The model uses the pattern it learned from the examples to answer the new question.


::::::::::::::::::::::::::::::::::::: challenge

### Challenge

To summarize this approach in a few steps, fill in the following gaps:
1.	Choose a Model: Select a **---** model from Hugging Face that suits your task.
   
2.	Load the Model: Use the **---** function to load the model and tokenizer.
  
3.	Craft Your Prompt: Write a **---** that includes one or more examples, depending on whether you’re doing one-shot or few-shot prompting.
  
4.	Generate Text: Call the **---** with your prompt to generate the **---**.
  
5.	Review the Output: Check the generated text to see if the model followed the **---** correctly.


:::::::::::::::::::::::: solution 

1.	Choose a Model: Select a **pre-trained** model from Hugging Face that suits your task.
   
2.	Load the Model: Use the **pipeline** function to load the model and tokenizer.
  
3.	Craft Your Prompt: Write a **prompt** that includes one or more examples, depending on whether you’re doing one-shot or few-shot prompting.
  
4.	Generate Text: Call the **generator** with your prompt to generate the **output**.
  
5.	Review the Output: Check the generated text to see if the model followed the **examples** correctly.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::::::: spoiler 

### Heads-up: Prompting Quality

Remember, the quality of the output heavily depends on the quality and relevance of the examples you provide. It’s also important to note that larger models tend to perform better at these tasks due to their greater capacity to understand and generalize from examples.

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints 

- Domain-specific LLMs are essential for tasks that require specialized knowledge.
- Prompt engineering, RAG, fine-tuning, and training from scratch are viable approaches to create DSLs.
- A mixed prompting-RAG approach is often preferred for its balance between performance and resource efficiency.
- Training from scratch offers the highest quality output but requires significant resources.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->

