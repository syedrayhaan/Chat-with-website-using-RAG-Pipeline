import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# 1. Website Scraper: Fetch and extract text from web pages
class WebsiteScraper:
    def __init__(self, url_list):
        self.url_list = url_list
        self.raw_text = ""

    def crawl_and_extract(self):
        print("Starting web scraping...")
        for url in self.url_list:
            print(f"Scraping: {url}")
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    page_text = soup.get_text()
                    self.raw_text += page_text + "\n"
                else:
                    print(f"Failed to fetch {url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        return self.raw_text

# 2. Text Segmenter: Split text into smaller chunks
class TextSegmenter:
    def __init__(self, raw_text):
        self.raw_text = raw_text

    def segment_text(self):
        """
        Segments the raw text into sentences using NLTK's PunktSentenceTokenizer.
        """
        print("Segmenting text into sentences...")
        
        # Punkt configuration to handle common abbreviations
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'e.g', 'i.e'])  
        
        # Initialize PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer(punkt_param)
        
        # Tokenize the text into sentences
        sentences = tokenizer.tokenize(self.raw_text)
        
        print(f"Total sentences segmented: {len(sentences)}")
        
        # Return text in manageable chunks (e.g., sentences grouped in chunks of 5)
        chunk_size = 5
        text_chunks = [
            " ".join(sentences[i:i + chunk_size]) 
            for i in range(0, len(sentences), chunk_size)
        ]
        return text_chunks

# 3. Embedding Generator: Generate vector embeddings for text chunks
class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks):
        print("Generating embeddings for text chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        print(f"Generated embeddings for {len(chunks)} chunks.")
        return embeddings

# 4. Embedding Store: Store embeddings in a FAISS index for retrieval
class EmbeddingStore:
    def __init__(self, dimension):
        print("Initializing FAISS index...")
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance

    def add_embeddings(self, embeddings):
        print("Adding embeddings to FAISS index...")
        self.index.add(np.array(embeddings))

    def save_index(self, file_path):
        faiss.write_index(self.index, file_path)
        print(f"FAISS index saved to: {file_path}")

    def load_index(self, file_path):
        self.index = faiss.read_index(file_path)
        print(f"FAISS index loaded from: {file_path}")

    def search_similar(self, query_embedding, k=5):
        print("Performing similarity search...")
        distances, indices = self.index.search(np.array(query_embedding), k)
        return indices

# 5. Query Processor: Process user queries and retrieve relevant chunks
class QueryProcessor:
    def __init__(self, embedding_model, faiss_index):
        self.model = embedding_model
        self.faiss_index = faiss_index

    def process_query(self, query):
        print("Generating embedding for user query...")
        query_embedding = self.model.encode([query])
        return query_embedding

    def retrieve_chunks(self, query_embedding, text_chunks, k=5):
        print("Retrieving relevant text chunks...")
        indices = self.faiss_index.search_similar(query_embedding, k)[0]
        # Convert NumPy array to a list of integers
        indices = indices.flatten().tolist()
        
        retrieved_chunks = [text_chunks[i] for i in indices if i < len(text_chunks)]
        return retrieved_chunks

# 6. Response Generator: Generate responses using a language model
class ResponseGenerator:
    def __init__(self, model_name="gpt2"):
        print("Loading language model...")
        self.llm = pipeline("text-generation", model=model_name)

    def generate_response(self, retrieved_chunks):
        print("Generating response from retrieved chunks...")
        context = " ".join(retrieved_chunks)
        response = self.llm(context, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']

# 7. Main Workflow
def main():
    # Step 1: Define the websites to scrape
    urls = ["https://www.stanford.edu/"]  # Replace with actual URLs

    # Step 2: Web scraping
    scraper = WebsiteScraper(urls)
    raw_text = scraper.crawl_and_extract()

    # Step 3: Text segmentation
    segmenter = TextSegmenter(raw_text)
    text_chunks = segmenter.segment_text()

    # Step 4: Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(text_chunks)

    # Step 5: Store embeddings
    embedding_store = EmbeddingStore(dimension=embeddings.shape[1])
    embedding_store.add_embeddings(embeddings)
    embedding_store.save_index("faiss_index.index")

    # Step 6: Load embeddings for query processing
    embedding_store.load_index("faiss_index.index")
    query_processor = QueryProcessor(embedding_generator.model, embedding_store)

    # Step 7: User query input
    user_query = input("Enter your query: ")
    query_embedding = query_processor.process_query(user_query)

    # Step 8: Retrieve relevant chunks
    retrieved_chunks = query_processor.retrieve_chunks(query_embedding, text_chunks)
    print(f"Retrieved Chunks: {retrieved_chunks}")

    # Step 9: Generate response
    response_generator = ResponseGenerator()
    response = response_generator.generate_response(retrieved_chunks)
    print("\nGenerated Response:\n", response)

if __name__ == "__main__":
    main()
