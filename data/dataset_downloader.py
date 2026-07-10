"""
Dataset Downloader and Preprocessor

This module downloads and prepares high-quality text datasets from reputable sources
for training the Predictive Text Input System.

Supported Datasets:
1. Project Gutenberg - Classic literature (https://www.gutenberg.org/)
2. Stanford Large Movie Review Dataset - IMDB reviews
3. Reuters-21578 - News articles
4. Brown Corpus - Balanced corpus of American English
5. Wikipedia articles - Encyclopedia content
"""

import os
import urllib.request
import zipfile
import tarfile
import json
from typing import List, Dict, Optional
import nltk
from nltk.corpus import brown, reuters
import requests
from tqdm import tqdm


class DatasetDownloader:
    """
    Downloads and prepares datasets from reputable academic sources.
    """
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset information for citations
        self.dataset_info = {
            "gutenberg": {
                "name": "Project Gutenberg",
                "url": "https://www.gutenberg.org/",
                "description": "Free eBooks of classic literature",
                "citation": "Project Gutenberg. (2021). Free eBooks. https://www.gutenberg.org/"
            },
            "imdb": {
                "name": "Stanford Large Movie Review Dataset",
                "url": "https://ai.stanford.edu/~amaas/data/sentiment/",
                "description": "50,000 movie reviews for sentiment analysis",
                "citation": "Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150)."
            },
            "reuters": {
                "name": "Reuters-21578",
                "url": "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html",
                "description": "Collection of Reuters newswire articles",
                "citation": "Lewis, D. D. (1997). Reuters-21578 text categorization collection. AT&T Labs Research."
            },
            "brown": {
                "name": "Brown Corpus",
                "url": "https://www1.essex.ac.uk/linguistics/external/clmt/w3c/corpus_ling/content/corpora/list/private/brown/brown.html",
                "description": "Balanced corpus of American English",
                "citation": "Francis, W. N., & Kucera, H. (1979). Brown corpus manual. Brown University."
            },
            "wiki": {
                "name": "Wikipedia Articles",
                "url": "https://dumps.wikimedia.org/",
                "description": "Encyclopedia articles from Wikipedia",
                "citation": "Wikimedia Foundation. (2024). Wikipedia: The Free Encyclopedia. https://en.wikipedia.org/"
            }
        }
    
    def download_gutenberg_books(self, book_ids: List[int] = None) -> str:
        """
        Download books from Project Gutenberg.
        
        Args:
            book_ids: List of book IDs to download (default: popular classics)
            
        Returns:
            Path to the combined text file
        """
        if book_ids is None:
            # Popular classic books
            book_ids = [
                1342,  # Pride and Prejudice - Jane Austen
                11,    # Alice's Adventures in Wonderland - Lewis Carroll
                74,    # The Adventures of Tom Sawyer - Mark Twain
                1661,  # The Adventures of Sherlock Holmes - Arthur Conan Doyle
                84,    # Frankenstein - Mary Shelley
                2701,  # Moby Dick - Herman Melville
                1400,  # Great Expectations - Charles Dickens
                98,    # A Tale of Two Cities - Charles Dickens
                345,   # Dracula - Bram Stoker
                76     # Adventures of Huckleberry Finn - Mark Twain
            ]
        
        print(f"Downloading {len(book_ids)} books from Project Gutenberg...")
        
        gutenberg_dir = os.path.join(self.data_dir, "gutenberg")
        os.makedirs(gutenberg_dir, exist_ok=True)
        
        all_text = []
        
        for book_id in tqdm(book_ids, desc="Downloading books"):
            try:
                url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
                filename = os.path.join(gutenberg_dir, f"book_{book_id}.txt")
                
                if not os.path.exists(filename):
                    urllib.request.urlretrieve(url, filename)
                
                with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    # Remove Project Gutenberg header/footer
                    text = self._clean_gutenberg_text(text)
                    all_text.append(text)
                    
            except Exception as e:
                print(f"Error downloading book {book_id}: {e}")
                continue
        
        # Combine all texts
        combined_path = os.path.join(gutenberg_dir, "gutenberg_corpus.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(all_text))
        
        print(f"Gutenberg corpus saved to: {combined_path}")
        print(f"Total size: {len(''.join(all_text))} characters")
        
        return combined_path
    
    def download_imdb_dataset(self) -> str:
        """
        Download Stanford IMDB Movie Review Dataset.
        
        Returns:
            Path to the processed text file
        """
        print("Downloading Stanford IMDB Movie Review Dataset...")
        
        imdb_dir = os.path.join(self.data_dir, "imdb")
        os.makedirs(imdb_dir, exist_ok=True)
        
        # Download URL
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        tar_path = os.path.join(imdb_dir, "aclImdb_v1.tar.gz")
        
        if not os.path.exists(tar_path):
            print("Downloading IMDB dataset (84MB)...")
            self._download_with_progress(url, tar_path)
        
        # Extract
        extract_path = os.path.join(imdb_dir, "aclImdb")
        if not os.path.exists(extract_path):
            print("Extracting dataset...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(imdb_dir)
        
        # Process reviews
        all_reviews = []
        
        for split in ['train', 'test']:
            for sentiment in ['pos', 'neg']:
                review_dir = os.path.join(extract_path, split, sentiment)
                if os.path.exists(review_dir):
                    for filename in os.listdir(review_dir):
                        if filename.endswith('.txt'):
                            filepath = os.path.join(review_dir, filename)
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                review = f.read().strip()
                                all_reviews.append(review)
        
        # Save combined reviews
        imdb_corpus_path = os.path.join(imdb_dir, "imdb_corpus.txt")
        with open(imdb_corpus_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(all_reviews))
        
        print(f"IMDB corpus saved to: {imdb_corpus_path}")
        print(f"Total reviews: {len(all_reviews)}")
        
        return imdb_corpus_path
    
    def prepare_nltk_corpora(self) -> Dict[str, str]:
        """
        Prepare NLTK corpora (Brown, Reuters).
        
        Returns:
            Dictionary mapping corpus names to file paths
        """
        print("Preparing NLTK corpora...")
        
        # Download NLTK data
        try:
            nltk.download('brown', quiet=True)
            nltk.download('reuters', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Note: Some NLTK downloads may have failed: {e}")
        
        nltk_dir = os.path.join(self.data_dir, "nltk")
        os.makedirs(nltk_dir, exist_ok=True)
        
        corpus_paths = {}
        
        # Brown Corpus
        try:
            print("Processing Brown Corpus...")
            brown_text = []
            for sent in brown.sents():
                brown_text.append(" ".join(sent))
            
            brown_path = os.path.join(nltk_dir, "brown_corpus.txt")
            with open(brown_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(brown_text))
            
            corpus_paths['brown'] = brown_path
            print(f"Brown corpus saved to: {brown_path}")
            
        except Exception as e:
            print(f"Error processing Brown corpus: {e}")
        
        # Reuters Corpus
        try:
            print("Processing Reuters Corpus...")
            reuters_text = []
            for fileid in reuters.fileids()[:1000]:  # Limit to first 1000 articles
                text = reuters.raw(fileid)
                reuters_text.append(text)
            
            reuters_path = os.path.join(nltk_dir, "reuters_corpus.txt")
            with open(reuters_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(reuters_text))
            
            corpus_paths['reuters'] = reuters_path
            print(f"Reuters corpus saved to: {reuters_path}")
            
        except Exception as e:
            print(f"Error processing Reuters corpus: {e}")
        
        return corpus_paths
    
    def download_wikipedia_sample(self, num_articles: int = 1000) -> str:
        """
        Download a sample of Wikipedia articles.
        
        Args:
            num_articles: Number of articles to download
            
        Returns:
            Path to the Wikipedia corpus file
        """
        print(f"Downloading {num_articles} Wikipedia articles...")
        
        wiki_dir = os.path.join(self.data_dir, "wikipedia")
        os.makedirs(wiki_dir, exist_ok=True)
        
        # Use Wikipedia API to get random articles
        articles = []
        
        for i in tqdm(range(num_articles), desc="Fetching articles"):
            try:
                # Get random article
                random_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
                response = requests.get(random_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    title = data.get('title', '')
                    extract = data.get('extract', '')
                    
                    if extract and len(extract) > 100:  # Only keep substantial articles
                        articles.append(f"Title: {title}\n\n{extract}")
                
                # Rate limiting
                if i % 10 == 0:
                    import time
                    time.sleep(0.1)
                    
            except Exception as e:
                continue
        
        # Save articles
        wiki_path = os.path.join(wiki_dir, "wikipedia_corpus.txt")
        with open(wiki_path, 'w', encoding='utf-8') as f:
            f.write("\n\n" + "="*50 + "\n\n".join(articles))
        
        print(f"Wikipedia corpus saved to: {wiki_path}")
        print(f"Total articles: {len(articles)}")
        
        return wiki_path
    
    def create_combined_corpus(self, datasets: List[str] = None, output_name: str = "combined_corpus.txt") -> str:
        """
        Create a combined corpus from multiple datasets.
        
        Args:
            datasets: List of dataset names to combine
            output_name: Name of the output file
            
        Returns:
            Path to the combined corpus
        """
        if datasets is None:
            datasets = ['gutenberg', 'brown', 'reuters']
        
        print(f"Creating combined corpus from: {', '.join(datasets)}")
        
        combined_texts = []
        
        for dataset in datasets:
            if dataset == 'gutenberg':
                if not os.path.exists(os.path.join(self.data_dir, "gutenberg", "gutenberg_corpus.txt")):
                    self.download_gutenberg_books()
                path = os.path.join(self.data_dir, "gutenberg", "gutenberg_corpus.txt")
            
            elif dataset == 'imdb':
                if not os.path.exists(os.path.join(self.data_dir, "imdb", "imdb_corpus.txt")):
                    self.download_imdb_dataset()
                path = os.path.join(self.data_dir, "imdb", "imdb_corpus.txt")
            
            elif dataset in ['brown', 'reuters']:
                nltk_paths = self.prepare_nltk_corpora()
                path = nltk_paths.get(dataset)
            
            elif dataset == 'wikipedia':
                if not os.path.exists(os.path.join(self.data_dir, "wikipedia", "wikipedia_corpus.txt")):
                    self.download_wikipedia_sample()
                path = os.path.join(self.data_dir, "wikipedia", "wikipedia_corpus.txt")
            
            else:
                print(f"Unknown dataset: {dataset}")
                continue
            
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    combined_texts.append(f"\n\n=== {dataset.upper()} CORPUS ===\n\n{text}")
                print(f"Added {dataset} corpus ({len(text)} characters)")
        
        # Save combined corpus
        combined_path = os.path.join(self.data_dir, output_name)
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(combined_texts))
        
        print(f"Combined corpus saved to: {combined_path}")
        print(f"Total size: {sum(len(text) for text in combined_texts)} characters")
        
        return combined_path
    
    def generate_citation_file(self) -> str:
        """
        Generate a citation file for the datasets used.
        
        Returns:
            Path to the citation file
        """
        citation_path = os.path.join(self.data_dir, "dataset_citations.txt")
        
        with open(citation_path, 'w', encoding='utf-8') as f:
            f.write("DATASET CITATIONS FOR PREDICTIVE TEXT INPUT SYSTEM\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("This project uses the following reputable datasets:\n\n")
            
            for dataset_id, info in self.dataset_info.items():
                f.write(f"{info['name']}\n")
                f.write(f"URL: {info['url']}\n")
                f.write(f"Description: {info['description']}\n")
                f.write(f"Citation: {info['citation']}\n")
                f.write("-" * 40 + "\n\n")
            
            f.write("ADDITIONAL NOTES:\n")
            f.write("- All datasets are publicly available and free to use for research\n")
            f.write("- NLTK corpora are academic-standard datasets\n")
            f.write("- Project Gutenberg texts are in the public domain\n")
            f.write("- Stanford datasets are widely used in academic research\n")
        
        print(f"Citation file saved to: {citation_path}")
        return citation_path
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers/footers."""
        lines = text.split('\n')
        
        # Find start of actual content (after headers)
        start_idx = 0
        for i, line in enumerate(lines):
            if '*** START OF' in line.upper() or 'CHAPTER' in line.upper():
                start_idx = i + 1
                break
        
        # Find end of actual content (before footers)
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if '*** END OF' in lines[i].upper():
                end_idx = i
                break
        
        return '\n'.join(lines[start_idx:end_idx])
    
    def _download_with_progress(self, url: str, filename: str):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)


def main():
    """Main function to demonstrate dataset downloading."""
    print("Dataset Downloader for Predictive Text Input System")
    print("=" * 60)
    
    downloader = DatasetDownloader()
    
    # Generate citation file
    downloader.generate_citation_file()
    
    print("\nAvailable datasets:")
    print("1. Project Gutenberg (Classic Literature)")
    print("2. Stanford IMDB Reviews")
    print("3. NLTK Brown Corpus")
    print("4. NLTK Reuters Corpus") 
    print("5. Wikipedia Sample")
    print("6. Combined Corpus")
    
    choice = input("\nWhich dataset would you like to download? (1-6, or 'all'): ").strip()
    
    if choice == '1':
        downloader.download_gutenberg_books()
    elif choice == '2':
        downloader.download_imdb_dataset()
    elif choice == '3' or choice == '4':
        downloader.prepare_nltk_corpora()
    elif choice == '5':
        downloader.download_wikipedia_sample()
    elif choice == '6':
        downloader.create_combined_corpus()
    elif choice.lower() == 'all':
        print("Downloading all datasets...")
        downloader.download_gutenberg_books()
        downloader.prepare_nltk_corpora()
        downloader.download_wikipedia_sample(500)  # Smaller sample for combined
        downloader.create_combined_corpus(['gutenberg', 'brown', 'reuters', 'wikipedia'])
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
