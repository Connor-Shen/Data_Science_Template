---
jupyter:
  title: Implement natural language processing with nltk
  dataset: Origin of Species Excerpt
  difficulty: Middle
  module: nltk
  idx: 10
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
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
import string
import matplotlib.pyplot as plt
```

### Step 2. Load the text and name it as "text"
```python
text_name = "Origin of Species Excerpt"
text = """
When we look to the individuals of the same variety or sub-variety of our older cultivated plants and animals, one of the first points which strikes us, is, that they generally differ much more from each other, than do the individuals of any one species or variety in a state of nature. But nothing is easy to specify cases where an animal or plant in a state of nature presents some slight abnormal structure, or instinct.
"""
```

### Step 3. Tokenize the text based on "word".
```python
tokens = word_tokenize(text)
```

### Step 4. Remove punctuation from the list of tokens
```python
tokens_no_punct = [word for word in tokens if word not in string.punctuation]
```

### Step 5. Remove stop words from the text using "english"
```python
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens_no_punct if word.lower() not in stop_words]
```

### Step 6. Convert all tokens to lowercase
```python
tokens_lowercase = [word.lower() for word in filtered_tokens]
```

### Step 7. Perform lemmatization to the list of tokens
```python
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens_lowercase]
```

### Step 8. Implement part-of-speech tagging for the above sentence using "pos_tag"
```python
pos_tagged_tokens = pos_tag(lemmatized_tokens)
```

### Step 9. Implement stemming for sentences after word segmentation using "porterstemmer"
```python
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
```

### Step 10. Extract named entities or perform chunking based on "ne_chunk"
```python
chunks = ne_chunk(pos_tagged_tokens)
```

### Step 11. Calculate frequency distribution of tokens.
```python
fdist = FreqDist(stemmed_tokens)
```

### Step 12. Display the "10" most common tokens.
```python
most_common_tokens = fdist.most_common(10)
print(most_common_tokens)
```

### Step 13. Visualize the frequency distribution using a bar chart with "Top 10 Frequent Words in Origin of Species Excerpt".
```python
fdist.plot(10, title="Top 10 Frequent Words in Origin of Species Excerpt")
```
