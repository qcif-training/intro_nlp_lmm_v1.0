---
title: 'Domain-Specific LLMs'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can tune the LLMs to be domain-specific?
- What are some available approaches to empower LLMs solve specific research problems?
- Which approach should I use for my research?
- What are the challenges and trade-offs of domain-specific LLMs?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Be able to identify approaches by which LLMs can be tuned for solving research problems.
- Be able to use introductory approaches for creating domain-specific LLMs.

::::::::::::::::::::::::::::::::::::::::::::::::

### 7.1.	Introduction to DSL (Available Approaches)

To enhance the response quality of an LLM for solving specific problems we need to use strategies by which we can tune the LLM. Generally, there are four ways to enhance the performance of LLMs:

#### 1. Prompt Optimization:
To elicit specific and accurate responses from LLMs by designing prompts strategically. 

**Zero-shot Prompting**: This is the simplest form of prompting where the LLM is given a task or question without any context or examples. It relies on the LLM’s pre-existing knowledge to generate a response. 

Example: “What is the capital of France?” The LLM would respond with “Paris” based on its internal knowledge. 

**Few-shot Prompting**: In this technique, the LLM is provided with a few examples to demonstrate the expected response format or content. 

Example: To determine sentiment, you might provide examples like “I love sunny days. (+1)” and “I hate traffic. (-1)” before asking the LLM to analyze a new sentence.

#### 2. Retrieval Augmented Generation (RAG):
To supplement the LLM’s generative capabilities with information retrieved from external databases or documents. 

**Retrieval**: The LLM queries a database to find relevant information that can inform its response. 

Example: If asked about recent scientific discoveries, the LLM might retrieve articles or papers on the topic. 

**Generation**: After retrieving the information, the LLM integrates it into a coherent response. 

Example: Using the retrieved scientific articles, the LLM could generate a summary of the latest findings in a particular field.

#### 3. Fine-Tuning: 
To adapt a general-purpose LLM to excel at a specific task or within a particular domain. 

**Language Modeling Task Fine-Tuning**: This involves training the LLM on a large corpus of text to improve its ability to predict the next word or phrase in a sentence. 

Example: An LLM fine-tuned on legal documents would become better at generating text that resembles legal writing. 

**Supervised Q&A Fine-Tuning**: Here, the LLM is trained on a dataset of question-answer pairs to enhance its performance on Q&A tasks.

Example: An LLM fine-tuned with medical Q&A pairs would provide more accurate responses to health-related inquiries.

#### 4.	Training from Scratch: 
Builds a model specifically for a domain, using relevant data from the ground up.


::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. 

Which approach do you think is more computation-intensive? Which is more accurate? How are these qualities related?  Evaluate the trade-offs between fine-tuning and other approaches.

![](fig/dsllms_1.png)

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. 

What is DSL and why are they useful for research tasks? Think of some examples of NLP tasks that require domain-specific LLMs, such as literature review, patent analysis, or material discovery. How do domain-specific LLMs improve the performance and accuracy of these tasks?

![](fig/dsllms_2.png)


::::::::::::::::::::::::::::::::::::::::::::::::

### 7.2.	Prompting

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

**Evaluate** the model’s performance using appropriate metrics and address any class imbalances with weighted loss functions or similar strategies. Save checkpoints to preserve progress and document every step of the fine-tuning process for transparency and reproducibility. 

**Ethical considerations** are paramount; strive for a model that is fair and unbiased. Ensure compliance with data protection regulations, especially when handling sensitive information. 

Lastly, manage **computational resources** wisely and engage with the Hugging Face community for additional support. Fine-tuning is iterative, and success often comes through continuous experimentation and learning.

::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. 

Guess the following architecture belongs to which optimization strategy:

![](fig/dsllms_3.png)

Figure. LLMs optimization (source)

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. 

What are the challenges and trade-offs of domain-specific LLMs, such as data availability, model size, and complexity? 

Consider some of the factors that affect the quality and reliability of domain-specific LLMs, such as the amount and quality of domain-specific data, the computational resources and time required for training or fine-tuning, and the generalization and robustness of the model. How do these factors pose problems or difficulties for domain-specific LLMs and how can we overcome them?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: Discussion

## Discuss in groups. 

What are some available approaches for creating domain-specific LLMs, such as fine-tuning and knowledge distillation? 

Consider some of the main steps and techniques for creating domain-specific LLMs, such as selecting a general LLM, collecting and preparing domain-specific data, training or fine-tuning the model, and evaluating and deploying the model. How do these approaches differ from each other and what are their advantages and disadvantages?

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

In this example, we provide the model with one translation example and then ask it to translate a new sentence. The model uses the context from the one-shot example to generate the translation. But what if we have a Few-Shot Prompting? Few-shot prompting gives the model several examples to learn from. This can improve the model’s ability to understand and complete the task. 

Here is how you can implement few-shot prompting:

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

## Challenge

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

## Heads-up: Prompting Quality

Remember, the quality of the output heavily depends on the quality and relevance of the examples you provide. It’s also important to note that larger models tend to perform better at these tasks due to their greater capacity to understand and generalize from examples.

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints 

- Domain-specific LLMs are essential for tasks that require specialized knowledge.
- Prompt engineering, RAG, fine-tuning, and training from scratch are viable approaches to create DSLs.
- A mixed prompting-RAG approach is often preferred for its balance between performance and resource efficiency.
- Training from scratch offers the highest quality output but requires significant resources.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
