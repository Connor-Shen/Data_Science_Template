---
jupyter:
  title: Implement natural language processing with nltk
  dataset: NLP
  difficulty: EASY
  module: nltk
  idx: 3
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
import matplotlib.pyplot as plt

```
### Step 2. Load the text and name it as "text"


```python

text_name = "Sample Text"
text = "Natural language processing (NLP) is a field of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human (natural) languages."

```

### Step 3. Tokenize the text based on "word".


```python

tokenizer = "word"
if tokenizer == "word":
    tokens = word_tokenize(text)
elif tokenizer == "sentence":
    tokens = sent_tokenize(text)

```

### Step 4. Remove punctuation from the list of tokens


```python

tokens = [word for word in tokens if word.isalpha()]

```

### Step 5. Remove stop words from the text using "english"


```python

stopwords_language = "english"
stop_words = set(stopwords.words(stopwords_language))
tokens = [word for word in tokens if word not in stop_words]
tokens

```



### Step 6. Convert all tokens to lowercase


```python

tokens = [word.lower() for word in tokens]
tokens

```


### Step 7. Perform lemmatization to the list of tokens


```python

lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]
tokens

```



### Step 8. Implement part-of-speech tagging for the above sentence using "pos_tag"


```python

tagger = "POS"
if tagger == "POS":
    tagged_tokens = pos_tag(tokens)
tagged_tokens

```



### Step 9. Implement stemming for sentences after word segmentation using "porterstemmer"


```python

stemmer = "porter"
if stemmer == "porter":
    stemmer = PorterStemmer()
elif stemmer == "lancaster":
    stemmer = LancasterStemmer()
tokens = [stemmer.stem(word) for word in tokens]
tokens

```




### Step 10. Extract named entities or perform chunking based on "ne_chunk"


```python

chunker = "NER"
if chunker == "NER":
    chunks = ne_chunk(tagged_tokens)
chunks

```


### Step 11. Calculate frequency distribution of tokens.

```python

freq_dist = FreqDist(tokens)
freq_dist

```

### Step 12. Display the "5" most common tokens.


```python

frequency_count = 5
common_tokens = freq_dist.most_common(frequency_count)
print(common_tokens)

```


### Step 13. Visualize the frequency distribution using a bar chart with "Top 10 Frequent Words in Origin of Species Excerpt".


```python

plot_title = "Token Frequency Distribution"
freq_dist.plot(frequency_count, title=plot_title)
plt.show()

```


