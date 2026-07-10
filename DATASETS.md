# Dataset Sources and Citations

## Overview
This document provides information about the datasets used in the Predictive Text Input System, all from reputable academic and public sources that you can confidently cite to your professor.

## Available Datasets

### 1. Project Gutenberg (Recommended for Literature)
- **Source**: https://www.gutenberg.org/
- **Description**: Free eBooks of classic literature in the public domain
- **Size**: ~60,000 free books
- **Quality**: High-quality, well-edited texts
- **Citation**: 
  ```
  Project Gutenberg. (2024). Free eBooks. Retrieved from https://www.gutenberg.org/
  ```
- **Sample Books Included**:
  - Pride and Prejudice - Jane Austen
  - Alice's Adventures in Wonderland - Lewis Carroll
  - The Adventures of Tom Sawyer - Mark Twain
  - Sherlock Holmes stories - Arthur Conan Doyle
  - Frankenstein - Mary Shelley
  - Moby Dick - Herman Melville

### 2. Stanford Large Movie Review Dataset (IMDB)
- **Source**: https://ai.stanford.edu/~amaas/data/sentiment/
- **Description**: 50,000 movie reviews for sentiment analysis
- **Size**: 84MB compressed
- **Quality**: Academic-grade dataset, widely used in research
- **Citation**:
  ```
  Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). 
  Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual 
  Meeting of the Association for Computational Linguistics: Human Language 
  Technologies (pp. 142-150).
  ```

### 3. Brown Corpus (NLTK)
- **Source**: https://www1.essex.ac.uk/linguistics/external/clmt/w3c/corpus_ling/content/corpora/list/private/brown/brown.html
- **Description**: Balanced corpus of American English from various genres
- **Size**: 1 million words
- **Quality**: Standard reference corpus in computational linguistics
- **Citation**:
  ```
  Francis, W. N., & Kucera, H. (1979). Brown Corpus Manual. 
  Brown University, Providence, Rhode Island.
  ```

### 4. Reuters-21578 Corpus (NLTK)
- **Source**: https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html
- **Description**: Collection of Reuters newswire articles
- **Size**: 21,578 articles
- **Quality**: Standard benchmark dataset for text classification
- **Citation**:
  ```
  Lewis, D. D. (1997). Reuters-21578 text categorization collection. 
  AT&T Labs Research.
  ```

### 5. Wikipedia Articles
- **Source**: https://dumps.wikimedia.org/
- **Description**: Encyclopedia articles from Wikipedia
- **Size**: Configurable (we use samples)
- **Quality**: Collaborative, well-structured content
- **Citation**:
  ```
  Wikimedia Foundation. (2024). Wikipedia: The Free Encyclopedia. 
  Retrieved from https://en.wikipedia.org/
  ```

## Academic Credibility

### Why These Datasets Are Academically Sound:

1. **Project Gutenberg**: 
   - Oldest digital library (since 1971)
   - Public domain texts ensure no copyright issues
   - Used in countless NLP research papers

2. **Stanford IMDB Dataset**:
   - Created by Stanford AI researchers
   - Peer-reviewed and published in ACL conference
   - Widely cited in academic literature (1000+ citations)

3. **NLTK Corpora**:
   - Standard corpora in computational linguistics
   - Distributed with NLTK, the leading NLP library
   - Used in university courses worldwide

4. **Reuters Corpus**:
   - Industry-standard benchmark dataset
   - Used in machine learning competitions
   - High-quality news content

5. **Wikipedia**:
   - Largest encyclopedia in human history
   - Multilingual and constantly updated
   - Used by major tech companies for AI training

## For Your Professor

### You can confidently tell your professor:

1. **"We use the Stanford IMDB dataset, which is a peer-reviewed academic dataset from Stanford University's AI lab, published in the ACL conference."**

2. **"Our literature training data comes from Project Gutenberg, the world's oldest digital library containing public domain texts."**

3. **"We also use the Brown Corpus and Reuters-21578, which are standard reference corpora in computational linguistics."**

4. **"All datasets are freely available, academically recognized, and widely used in NLP research."**

### Research Papers Using These Datasets:

- **IMDB Dataset**: Used in 1000+ research papers
- **Brown Corpus**: Referenced in linguistics research since 1960s
- **Reuters**: Standard benchmark in text classification
- **Project Gutenberg**: Used in style analysis, authorship attribution

## Dataset Statistics

| Dataset | Size | Domain | Language | Quality |
|---------|------|--------|----------|---------|
| Project Gutenberg | ~4GB | Literature | English | Very High |
| IMDB Reviews | 84MB | Movie Reviews | English | High |
| Brown Corpus | 15MB | Mixed Genres | English | Very High |
| Reuters | 27MB | News | English | High |
| Wikipedia Sample | Variable | Encyclopedia | English | High |

## How to Download

Run the dataset downloader:
```bash
python data/dataset_downloader.py
```

Or download specific datasets:
```python
from data.dataset_downloader import DatasetDownloader

downloader = DatasetDownloader()
downloader.download_gutenberg_books()      # Literature
downloader.download_imdb_dataset()         # Reviews  
downloader.prepare_nltk_corpora()          # Brown & Reuters
downloader.create_combined_corpus()        # All combined
```

## License and Usage

- **Project Gutenberg**: Public Domain
- **IMDB Dataset**: Academic Use (Stanford License)
- **NLTK Corpora**: Academic/Research Use
- **Wikipedia**: Creative Commons

All datasets are appropriate for academic projects and research.
