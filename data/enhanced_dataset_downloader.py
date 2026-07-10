"""
Enhanced Dataset Downloader with More Diverse Sources

This version includes:
1. More Project Gutenberg books (50+ classic books)
2. Academic writing samples
3. Conversational text data
4. Domain-specific datasets
"""

from .dataset_downloader import DatasetDownloader
import os
import requests
import json
from typing import List, Dict
from tqdm import tqdm

class EnhancedDatasetDownloader(DatasetDownloader):
    """Enhanced dataset downloader with more diverse sources."""
    
    def __init__(self, data_dir: str = "datasets"):
        super().__init__(data_dir)
        
        # Additional Gutenberg books (50+ classic books)
        self.extended_book_ids = [
            1342,   # Pride and Prejudice
            11,     # Alice in Wonderland
            74,     # Tom Sawyer
            1661,   # Sherlock Holmes
            84,     # Frankenstein
            2701,   # Moby Dick
            1400,   # Great Expectations
            98,     # Tale of Two Cities
            345,    # Dracula
            76,     # Huckleberry Finn
            16,     # Peter Pan
            1952,   # The Yellow Wallpaper
            1064,   # The Odyssey
            158,    # Emma
            174,    # The Picture of Dorian Gray
            1232,   # The Prince
            215,    # The Call of the Wild
            244,    # A Study in Scarlet
            2554,   # The Hound of the Baskervilles
            120,    # Treasure Island
            33,     # The Scarlet Letter
            1080,   # A Modest Proposal
            805,    # This Side of Paradise
            768,    # Wuthering Heights
            41,     # The Legend of Sleepy Hollow
            730,    # Oliver Twist
            1184,   # The Count of Monte Cristo
            145,    # Middlemarch
            25344,  # The Scarlet Pimpernel
            2600,   # War and Peace
            408,    # The Souls of Black Folk
            1260,   # Jane Eyre
            219,    # Heart of Darkness
            1497,   # Republic
            3207,   # Leviathan
            16328,  # Beowulf
            2814,   # Dubliners
            42,     # The Strange Case of Dr. Jekyll and Mr. Hyde
            236,    # The Jungle Book
            2852,   # The Hunchback of Notre Dame
            55,     # The Wonderful Wizard of Oz
            160,    # The Awakening
            35,     # The Time Machine
            36,     # The War of the Worlds
            43,     # The Strange Case of Dr. Jekyll and Mr. Hyde
            209,    # The Turn of the Screw
            121,    # Northanger Abbey
            1399,   # Anna Karenina
            161,    # Sense and Sensibility
            32,     # Herland
            768,    # Wuthering Heights
            203,    # Uncle Tom's Cabin
            2701    # Moby Dick
        ]
    
    def download_enhanced_gutenberg(self) -> str:
        """Download an enhanced set of Gutenberg books."""
        return self.download_gutenberg_books(self.extended_book_ids)
    
    def download_academic_samples(self) -> str:
        """
        Download academic writing samples from arXiv and other sources.
        Returns path to academic corpus.
        """
        print("Downloading academic writing samples...")
        
        academic_dir = os.path.join(self.data_dir, "academic")
        os.makedirs(academic_dir, exist_ok=True)
        
        # Use arXiv API to get abstracts
        samples = []
        categories = ['cs.AI', 'cs.CL', 'cs.LG', 'stat.ML']
        
        for category in categories:
            try:
                url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results=100"
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Parse XML response (simplified)
                    text = response.text
                    abstracts = [
                        text[text.find("<abstract>") + 10:text.find("</abstract>")]
                        for text in text.split("<entry>")[1:]
                    ]
                    samples.extend(abstracts)
            
            except Exception as e:
                print(f"Error downloading academic samples: {e}")
                continue
        
        academic_path = os.path.join(academic_dir, "academic_corpus.txt")
        with open(academic_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(samples))
        
        return academic_path
    
    def download_conversational_data(self) -> str:
        """
        Download conversational text data from various sources.
        Returns path to conversational corpus.
        """
        print("Preparing conversational text data...")
        
        conv_dir = os.path.join(self.data_dir, "conversational")
        os.makedirs(conv_dir, exist_ok=True)
        
        # Use sample conversations from multiple sources
        conversations = [
            # General conversation samples
            "How are you doing today? I'm doing well, thank you for asking.",
            "Would you like to grab coffee sometime? Sure, that sounds great!",
            
            # Technical discussions
            "Could you help me with this code? What seems to be the problem?",
            "I'm getting an error when I run the script. Can you show me the error message?",
            
            # Academic discussions
            "The results suggest a strong correlation between the variables.",
            "We should consider alternative hypotheses before drawing conclusions.",
            
            # Common phrases
            "It's nice to meet you. Likewise!",
            "Thanks for your help. You're welcome!",
            
            # Add many more conversation samples...
        ]
        
        conv_path = os.path.join(conv_dir, "conversational_corpus.txt")
        with open(conv_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(conversations))
        
        return conv_path
    
    def create_domain_specific_datasets(self) -> Dict[str, str]:
        """
        Create domain-specific datasets for better specialized predictions.
        Returns dict mapping domain names to corpus paths.
        """
        print("Creating domain-specific datasets...")
        
        domain_dir = os.path.join(self.data_dir, "domains")
        os.makedirs(domain_dir, exist_ok=True)
        
        domains = {
            "technical": [
                "The code implements a neural network using PyTorch.",
                "Debug the error in the function call.",
                # Add more technical content...
            ],
            "academic": [
                "The study examines the correlation between variables.",
                "The results suggest a significant effect.",
                # Add more academic content...
            ],
            "business": [
                "The quarterly report shows strong growth.",
                "Please review the attached proposal.",
                # Add more business content...
            ],
            "creative": [
                "The sunset painted the sky in vibrant colors.",
                "Her eyes sparkled with determination.",
                # Add more creative content...
            ]
        }
        
        domain_paths = {}
        
        for domain, texts in domains.items():
            domain_path = os.path.join(domain_dir, f"{domain}_corpus.txt")
            with open(domain_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(texts))
            domain_paths[domain] = domain_path
        
        return domain_paths
    
    def create_enhanced_combined_corpus(self) -> str:
        """
        Create a combined corpus from all enhanced datasets.
        Returns path to enhanced combined corpus.
        """
        print("Creating enhanced combined corpus...")
        
        enhanced_path = os.path.join(self.data_dir, "enhanced_corpus.txt")
        if os.path.exists(enhanced_path):
            print(f"Using existing enhanced corpus: {enhanced_path}")
            return enhanced_path
            
        # Check for existing datasets before downloading
        gutenberg_path = os.path.join(self.data_dir, "gutenberg", "gutenberg_corpus.txt")
        academic_path = os.path.join(self.data_dir, "academic", "academic_corpus.txt")
        conv_path = os.path.join(self.data_dir, "conversational", "conversational_corpus.txt")
        wiki_path = os.path.join(self.data_dir, "wikipedia", "wikipedia_corpus.txt")
        
        # Download only missing datasets
        if not os.path.exists(gutenberg_path):
            gutenberg_path = self.download_enhanced_gutenberg()
        if not os.path.exists(academic_path):
            academic_path = self.download_academic_samples()
        if not os.path.exists(conv_path):
            conv_path = self.download_conversational_data()
        if not os.path.exists(wiki_path):
            wiki_path = self.download_wikipedia_sample(num_articles=2000)
            
        # Create domain datasets if needed
        domain_paths = self.create_domain_specific_datasets()
        
        # Prepare NLTK corpora if needed
        nltk_dir = os.path.join(self.data_dir, "nltk")
        if not os.path.exists(nltk_dir):
            nltk_paths = self.prepare_nltk_corpora()
        else:
            nltk_paths = {
                'brown': os.path.join(nltk_dir, "brown_corpus.txt"),
                'reuters': os.path.join(nltk_dir, "reuters_corpus.txt")
            }
        
        # Combine all texts with clear section markers
        all_texts = []
        
        # Add Gutenberg texts
        with open(gutenberg_path, 'r', encoding='utf-8') as f:
            all_texts.append("=== CLASSIC LITERATURE ===\n\n" + f.read())
        
        # Add academic texts
        with open(academic_path, 'r', encoding='utf-8') as f:
            all_texts.append("=== ACADEMIC WRITING ===\n\n" + f.read())
        
        # Add conversational data
        with open(conv_path, 'r', encoding='utf-8') as f:
            all_texts.append("=== CONVERSATIONAL TEXT ===\n\n" + f.read())
        
        # Add domain-specific texts
        for domain, path in domain_paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                all_texts.append(f"=== {domain.upper()} DOMAIN ===\n\n" + f.read())
        
        # Add NLTK corpora
        for corpus, path in nltk_paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                all_texts.append(f"=== {corpus.upper()} CORPUS ===\n\n" + f.read())
        
        # Add Wikipedia text
        with open(wiki_path, 'r', encoding='utf-8') as f:
            all_texts.append("=== WIKIPEDIA ARTICLES ===\n\n" + f.read())
        
        # Save enhanced combined corpus
        enhanced_path = os.path.join(self.data_dir, "enhanced_corpus.txt")
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            f.write("\n\n" + "="*50 + "\n\n".join(all_texts))
        
        print(f"Enhanced combined corpus saved to: {enhanced_path}")
        print(f"Total size: {sum(len(text) for text in all_texts)} characters")
        
        return enhanced_path


def main():
    """Main function to demonstrate enhanced dataset downloading."""
    print("Enhanced Dataset Downloader for Predictive Text Input System")
    print("=" * 60)
    
    downloader = EnhancedDatasetDownloader()
    
    # Generate citation file
    downloader.generate_citation_file()
    
    print("\nCreating enhanced dataset collection...")
    enhanced_path = downloader.create_enhanced_combined_corpus()
    
    print("\nDataset creation complete!")
    print(f"Enhanced corpus saved to: {enhanced_path}")


if __name__ == "__main__":
    main()