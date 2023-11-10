---
jupyter:
  title: Implement natural language processing with nltk
  dataset: AI
  difficulty: Middle
  module: nltk
  idx: 2
---

### Step 1. Import required python packages.

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
```

### Step 2. Load the text and name it as "ai_text"
```python
ai_text = "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. The term can also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal."
```

### Step 3. Tokenize the text based on "word_tokenizer".
```python
tokens = word_tokenize(ai_text)
```

### Step 4. Remove punctuation from the list of tokens.
```python
tokens = [word for word in tokens if word.isalnum()]
```

### Step 5. Remove stop words from the text using "english".
```python
stop_words = set(stopwords.words("english"))
tokens = [word for word in tokens if word not in stop_words]
```

### Step 6. Convert all tokens to lowercase.
```python
tokens = [word.lower() for word in tokens]
```

### Step 7. Perform lemmatization to the list of tokens.
```python
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]
```

### Step 8. Implement part-of-speech tagging for the above sentence using "POS_tagger".
```python
tagged_tokens = pos_tag(tokens)
```

### Step 9. Implement stemming for sentences after word segmentation using "lancaster_stemmer".
```python
stemmer = LancasterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens]
```

### Step 10. Extract named entities or perform chunking based on "named_entity_chunker".
```python
chunked_sentences = ne_chunk(tagged_tokens)
```

### Step 11. Calculate frequency distribution of tokens.
```python
freq_dist = FreqDist(tokens)
```

### Step 12. Display the 10 most common tokens.
```python
common_tokens = freq_dist.most_common(10)
```

### Step 13. Visualize the frequency distribution using a bar chart with "Artificial Intelligence Insights".
```python
plt.figure(figsize=(12, 6))
freq_dist.plot(10, title="Artificial Intelligence Insights")
plt.show()
```
