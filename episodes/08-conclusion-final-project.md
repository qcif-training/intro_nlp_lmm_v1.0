---
title: 'Wrap-up and Final Project'
teaching: 10
exercises: 1
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are the core concepts and techniques we’ve learned about NLP and LLMs?
- How can these techniques be applied to solve real-world problems?
- What are the future directions and opportunities in NLP?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- To be able to synthesize the key concepts from each episode.
- To plan a path for further learning and exploration in NLP and LLMs.

::::::::::::::::::::::::::::::::::::::::::::::::


## 8.1.	Takeaway from This Workshop

We have covered a vast landscape of NLP, starting with the basics and moving towards the intricacies of LLMs. Here is a brief recap to illustrate our journey: 

- **Text Preprocessing**: Imagine cleaning a dataset of tweets for sentiment analysis. We learned how to remove noise and prepare the text for accurate classification.
- **Text Analysis**: Consider the task of extracting key information from news articles. Techniques like Named Entity Recognition helped us identify and categorize entities within the text.
- **Word Embedding**: We explored how words can be converted into vectors, enabling us to capture semantic relationships, as seen in the Word2Vec algorithm.
- **Transformers and LLMs**: We saw how transformers like BERT and GPT can be fine-tuned for tasks such as summarizing medical research papers and showcasing their power and flexibility.


::::::::::::::::::::::::::::::::::::: challenge

### Quiz


**A)** Gensim

**B)** Word2Vec	

**C)** Named Entity Recognition	

**D)** Part-of-Speech Tagging	

**E)** Stop-words Removal	

**F)** Transformers	

**G)** Bag of Words	

**H)** Tokenization	

**I)** BERT	

**J)** Lemmatization



**1.  A masked language model for NLP tasks that require a good contextual understanding of an entire sequence.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**2. “A process of reducing words to their root form, enabling the analysis of word frequency.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**3. An algorithm that uses neural networks to understand the relationships and meanings in human language.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**4. A technique for identifying the parts of speech for each word in a given sentence.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**5. A process where an algorithm takes a string of text and identifies relevant nouns that are mentioned in that string.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**6. A library that provides tools for machine learning and statistical modeling.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**7. A model that predicts the next word in a sentence based on the words that come before it.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**8. One of the first steps in text analysis, which involves breaking down text into individual elements.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**9. A technique that groups similar words together in vector space.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**10. A method for removing commonly used words that carry little meaning.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



:::::::::::::::: solution

A:

**1.  A masked language model for NLP tasks that require a good contextual understanding of an entire sequence.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [X] I - [ ] J



**2. “A process of reducing words to their root form, enabling the analysis of word frequency.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [X] J



**3. An algorithm that uses neural networks to understand the relationships and meanings in human language.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [X] F - [ ] G - [ ] H - [ ] I - [ ] J



**4. A technique for identifying the parts of speech for each word in a given sentence.**

[ ] A - [ ] B - [ ] C - [X] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**5. A process where an algorithm takes a string of text and identifies relevant nouns that are mentioned in that string.**

[ ] A - [ ] B - [X] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**6. A library that provides tools for machine learning and statistical modeling.**

[X] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**7. A model that predicts the next word in a sentence based on the words that come before it.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [X] F - [ ] G - [ ] H - [ ] I - [ ] J



**8. One of the first steps in text analysis, which involves breaking down text into individual elements.**

[ ] A - [ ] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [X] H - [ ] I - [ ] J



**9. A technique that groups similar words together in vector space.**

[ ] A - [X] B - [ ] C - [ ] D - [ ] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J



**10. A method for removing commonly used words that carry little meaning.**

[ ] A - [ ] B - [ ] C - [ ] D - [X] E - [ ] F - [ ] G - [ ] H - [ ] I - [ ] J

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::




::::::::::::::::::::::::::::::::::::: challenge

### Discussion


**Field of Interest**

Teamwork: Share insights on how NLP can be applied in your field of interest.


:::::::::::::::: solution

***Environmental Science***

- NLP for Climate Change Research: How can NLP help in analyzing large volumes of research papers on climate change to identify trends and gaps in the literature?
- Social Media Analysis for Environmental Campaigns: Discuss the use of sentiment analysis to gauge public opinion on environmental policies.
- Automating Environmental Compliance: Share insights on how NLP can streamline the process of checking compliance with environmental regulations in corporate documents.

***Education***

- Personalized Learning: Explore the potential of NLP in creating personalized learning experiences by analyzing student feedback and performance.
- Content Summarization: Discuss the benefits of using NLP to summarize educational content for quick revision.
- Language Learning: Share thoughts on the role of NLP in developing language learning applications that adapt to the learner’s proficiency level.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: challenge

## Mini-Project: Build and Optimize a DSL Question-Answering System

From the Hugging Face model hub, use the **GPT-2** model to build a question-answering system that can answer questions specific to a particular field (e.g., environmental science).
I. Test your model using the zero-shot prompting technique.
II. Test your model using the few-shot prompting technique. Ask the same question as in the zero-shot test and compare the generated answers.

:::::::::::::::::::::::: solution 

A.I: 

Context Example: Environmental science and climate change

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Since GPT-2 does not have a pad token, we set it to the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Function to generate a response from the model
def get_model_response(question):
    # Encode the prompt with an attention mask and padding
    encoded_input = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        return_tensors='pt',
        padding='max_length',  # Pad to max length
        truncation=True,
        max_length=100
    )
    
    # Generate the response with the attention mask and pad token id
    outputs = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        pad_token_id=tokenizer.eos_token_id,
        max_length=200,
        num_return_sequences=1
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

```

Now, we need to create a function that takes user inputs and provides answers:

```python

# Main function to interact with the user
def main():
    # Ask the user for their question
    user_question = input("What is your question about climate change? ")
    
    # Get the model's response
    answer = get_model_response(user_question)
    
    # Print the answer
    print(f"AI: {answer}")

# Run the main function
if __name__ == "__main__":
    main()

```

For the second part of the project, We will use few-shot prompting by providing examples of questions and answers related to environmental topics.

A.II: 

```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the pad token to the eos token for the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Function to generate a response from the model
def get_model_response(question):
    # Construct the prompt with clear instructions and examples
    prompt = f"""
    I am an AI trained to answer questions about climate change. Here are some examples:
    Q: What causes climate change?
    A: Climate change is primarily caused by the accumulation of greenhouse gases in the atmosphere due to human activities such as burning fossil fuels and deforestation.
    Q: How does deforestation affect climate change?
    A: Deforestation leads to increased carbon dioxide levels in the atmosphere because trees that absorb carbon dioxide are removed.
    Q: What is climate change?
    A: Climate change refers to significant changes in global temperatures and weather patterns over time.
    Q: What causes global warming?
    A: Global warming is primarily caused by the increase of greenhouse gases like carbon dioxide in the atmosphere due to human activities.
    Q: How does deforestation affect climate change?
    A: Deforestation contributes to climate change by reducing the number of trees that can absorb carbon dioxide, a greenhouse gas.
    Q: Can planting trees help combat climate change?
    A: Yes, planting trees can help mitigate climate change as trees absorb carbon dioxide from the atmosphere.
    Q: What is renewable energy?
    A: Renewable energy is derived from natural processes that are replenished constantly, like wind or solar power.
    Q: Why is conserving water important for the environment?
    A: Conserving water helps protect ecosystems, saves energy, and reduces the impact on water resources.
    Q: What is sustainable living?
    A: Sustainable living involves reducing one's carbon footprint by altering transportation methods, energy consumption, and diet.
    Q: How do electric cars reduce pollution?
    A: Electric cars produce fewer greenhouse gases and air pollutants than conventional vehicles.
    Q: What is the impact of climate change on wildlife?
    A: Climate change can alter habitats and food sources, leading to species migration and possible extinction.
    Q: How does recycling help the environment?
    A: Recycling reduces the need for raw materials, saves energy, and decreases pollution.
    Q: What is carbon footprint?
    A: A carbon footprint is the total amount of greenhouse gases emitted by an individual, organization, event, or product.
    Q: Why is biodiversity important for the ecosystem?
    A: Biodiversity boosts ecosystem productivity where each species has an important role to play.
    Q: What are the effects of ocean acidification?
    A: Ocean acidification can lead to the weakening of coral reefs and impact shell-forming marine life, disrupting the ocean's balance.
    Q: How does climate change affect agriculture?
    A: Climate change can alter crop yield, reduce water availability, and increase pests and diseases.
    Q: What is the Paris Agreement?
    A: The Paris Agreement is an international treaty aimed at reducing carbon emissions to combat climate change.
    Q: How do fossil fuels contribute to global warming?
    A: Burning fossil fuels releases large amounts of CO2, a major greenhouse gas, into the atmosphere.
    Q: What is the significance of the ozone layer?
    A: The ozone layer absorbs most of the sun's harmful ultraviolet radiation, protecting living organisms on Earth.
    Q: What are green jobs?
    A: Green jobs are positions in businesses that contribute to preserving or restoring the environment.
    Q: How can we make our homes more energy-efficient?
    A: We can insulate our homes, use energy-efficient appliances, and install smart thermostats to reduce energy consumption.
    Q: What is the role of the United Nations in climate change?
    A: The United Nations facilitates global climate negotiations and helps countries develop and implement climate policies.
    Q: What are climate change mitigation and adaptation?
    A: Mitigation involves reducing the flow of greenhouse gases into the atmosphere, while adaptation involves adjusting to current or expected climate change.
    Q: How does urbanization affect the environment?
    A: Urbanization can lead to habitat destruction, increased pollution, and higher energy consumption.
    Q: What is a carbon tax?
    A: A carbon tax is a fee imposed on the burning of carbon-based fuels, aimed at reducing greenhouse gas emissions.
    Q: How does air pollution contribute to climate change?
    A: Certain air pollutants like methane and black carbon have a warming effect on the atmosphere.
    Q: What is an ecological footprint?
    A: An ecological footprint measures the demand on Earth's ecosystems and compares it to nature's ability to regenerate resources.
    Q: What are sustainable development goals?
    A: Sustainable development goals are a collection of 17 global goals set by the United Nations to end poverty, protect the planet, and ensure prosperity for all.
    Q: How does meat consumption affect the environment?
    A: Meat production is resource-intensive and contributes to deforestation, water scarcity, and greenhouse gas emissions.
    Q: What is an endangered species?
    A: An endangered species is a type of organism that is at risk of extinction due to a drastic decline in population.
    Q: How can businesses reduce their environmental impact?
    A: Businesses can implement sustainable practices, reduce waste, use renewable energy, and invest in eco-friendly technologies.
    Q: What is a climate model?
    A: A climate model is a computer simulation used to predict future climate patterns based on different environmental scenarios.
    Q: Why are wetlands important for the environment?
    A: Wetlands provide critical habitat for many species, store floodwaters, and maintain surface water flow during dry periods.
    Q: What is geoengineering?
    A: Geoengineering is the deliberate large-scale intervention in the Earth’s climate system, aimed at mitigating the adverse effects of climate change.
    Q: How does plastic pollution affect marine life?
    A: Plastic pollution can injure or poison marine wildlife and disrupt marine ecosystems through ingestion and entanglement.
    Q: What is a carbon sink?
    A: A carbon sink is a natural or artificial reservoir that accumulates and stores carbon-containing chemical compounds for an indefinite period.
    Q: How do solar panels work?
    A: Solar panels convert sunlight into electricity through photovoltaic cells.
    Q: What is the impact of climate change on human health?
    A: Climate change can lead to health issues like heatstroke, allergies, and diseases spread by mosquitoes and ticks.
    Q: What is a green economy?
    A: A green economy is an economy that aims for sustainable development without degrading the environment.
    Q: How does energy consumption contribute to climate change?
    A: High energy consumption, especially from non-renewable sources, leads to higher emissions of greenhouse gases.
    Q: What is the Kyoto Protocol?
    A: The Kyoto Protocol is an international treaty that commits its parties to reduce greenhouse gas emissions.
    Q: How can we protect coastal areas from rising sea levels?
    A: We can protect coastal areas by restoring mangroves, building sea walls, and implementing better land-use policies.
    Q: What is a heatwave, and how is it linked to climate change?
    A: A heatwave is a prolonged period of excessively hot weather, which may become more frequent and intense due to climate change.
    Q: How does climate change affect water resources?
    A: Climate change can lead to changes in precipitation patterns, reduced snowpack, and increased evaporation rates.
    Q: What is a carbon credit?
    A: A carbon credit is a permit that allows the holder to emit a certain amount of carbon dioxide or other greenhouse gases.
    Q: What are the benefits of wind energy?
    A: Wind energy is a clean, renewable resource that reduces reliance on fossil fuels and decreases greenhouse gas emissions.
    Q: What is an energy audit?
    A: An energy audit is an assessment of energy use in a home or business to identify ways to improve efficiency and reduce costs.
    Q: How do wildfires contribute to climate change?
    A: Wildfires release stored carbon dioxide into the atmosphere and reduce the number of trees that can absorb CO2.
    Q: What is a sustainable diet?
    A: A sustainable diet involves consuming food that is healthy for individuals and sustainable for the environment.
    Q: How does climate change affect the polar regions?
    A: Climate change leads to melting ice caps, which can result in rising sea levels and loss of habitat for polar species.
    Q: What is the role of youth in climate action?
    A: Youth can play a crucial role in climate action through advocacy, innovation, and leading by example in sustainable practices.
    Q: What is the significance of Earth Day?
    A: Earth Day is an annual event to demonstrate support for environmental protection and promote awareness of environmental issues.

    Q: {question}
    A:"""
    
    # Encode the prompt with an attention mask and padding
    encoded_input = tokenizer.encode_plus(
        prompt,
        add_special_tokens=True,
        return_tensors='pt',
        padding='max_length',  # Pad to max length
        truncation=True,
        max_length=100
    )
    
    # Generate the response with the attention mask and pad token id
    outputs = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        pad_token_id=tokenizer.eos_token_id,
        max_length=200,
        num_return_sequences=1
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split('A:')[1].strip()  # Extract the AI's response

```

Similar to the first part of the project, we need to create a user interface function:

```python

def main():
    user_question = input("Please enter your question about climate change: ")
    
    # Get the model's response
    answer = get_model_response(user_question)
    
    # Print the answer
    print(f"AI: {answer}")

# Run the main function
if __name__ == "__main__":
    main()

```



The model should provide a more relevant answer based on the few-shot examples provided. In this challenge, we used the **GPT-2** model from Hugging Face’s transformers library to create a question-answering system. Few-shot prompting is employed to give the model context about environmental topics, which helps it generate more accurate answers to user queries. 

However, you should note that the performance enhancement is not impressive sometimes and it may need fine-tuning to get more accurate responses.

:::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::



## 8.2.	Further Resources

For continued learning, here are detailed resources:

- *Natural Language Processing Specialization (Coursera)*: A series of courses that cover NLP foundations, algorithms, and how to build NLP applications.
- *Stanford NLP Group*: Access to pioneering NLP research, datasets, and tools like Stanford Parser and Stanford POS Tagger.
- *Hugging Face*: A platform for sharing and collaborating on ML models, with a focus on democratizing NLP technologies.
- *Kaggle*: An online community for data scientists, offering datasets, notebooks, and competitions to practice and improve your NLP skills.
  

Each resource is a gateway to further knowledge, community engagement, and hands-on experience.


## 8.3.	Feedback

Please help us improve by answering the following survey questions:

**1.	How would you rate the overall quality of the workshop?**
   
[ ] Excellent,  [ ] Good,  [ ] Average,  [ ] Below Average,  [ ] Poor



**2.	Was the pace of the workshop appropriate?**
   
[ ] Too fast,     [ ] Just right,     [ ] Too slow



**3.	How clear were the instructions and explanations?**
   
[ ] Very clear,     [ ] Clear,     [ ] Somewhat clear,      [ ] Not clear



**4.	What was the most valuable part of the workshop for you?**

   

**5.	How can we improve the workshop for future participants?**



*Your feedback is crucial for us to evolve and enhance the learning experience.*


::::::::::::::::::::::::::::::::::::: keypoints 

- Various NLP techniques from preprocessing to advanced LLMs are reviewed.
- NLPs’ transformative potential provides real-world applications in diverse fields.
- Few-shot learning can enhance the performance of LLMs for specific fields of research.
- Valuable resources are highlighted for continued learning and exploration in the field of NLP.

::::::::::::::::::::::::::::::::::::::::::::::::
