---
jupyter:
  title: Implement natural language processing with nltk
  dataset: Shakespeare
  difficulty: Middle
  module: nltk
  idx: 8
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
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
```

### Step 2. Load the text and name it as "new_text_sample"
```python
new_text_sample = "Romeo and Juliet is a tragedy written by William Shakespeare. It is among the most popular plays ever written in the English language. The play revolves around two young star-crossed lovers whose deaths ultimately reconcile their feuding families."
```

### Step 3. Tokenize the text based on "sentence_tokenizer".
```python
sentences = sent_tokenize(new_text_sample)
```

### Step 4. Remove punctuation from the list of sentences.
```python
cleaned_sentences = [' '.join(word for word in word_tokenize(sentence) if word.isalnum()) for sentence in sentences]
```

### Step 5. Remove stop words from the text using "english".
```python
stop_words = set(stopwords.words("english"))
tokens = [word for word in word_tokenize(' '.join(cleaned_sentences)) if word not in stop_words]
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

### Step 12. Display the 15 most common tokens.
```python
common_tokens = freq_dist.most_common(15)
```

### Step 13. Visualize the frequency distribution using a bar chart with "Shakespeare's Play Analysis".
```python
plt.figure(figsize=(12, 6))
freq_dist.plot(15, title="Shakespeare's Play Analysis")
plt.show()
```
