from .models import ImageEmbeddingModel
from .preprocessing import ImagePreprocessor
from .similarity import SimilaritySearch


class CBIRSystem:
    def __init__(self):
        self.model = ImageEmbeddingModel()
        self.preprocessor = ImagePreprocessor()
        self.similarity_search = SimilaritySearch(self.model, self.preprocessor)
    
    def process_dataset(self, folder_path):
        print("[INFO] Extracting embeddings for dataset...")
        return self.similarity_search.compute_folder_embeddings(folder_path)
    
    def process_query(self, query_path):
        print("[INFO] Extracting embedding for query image...")
        query_tensor = self.preprocessor.preprocess_image(query_path)
        return self.preprocessor.get_embedding(self.model.model, query_tensor, self.model.get_device())
    
    def find_similar(self, query_embedding, all_embeddings, image_paths, top_k):
        print("[INFO] Finding most similar images...")
        return self.similarity_search.find_similar_images(query_embedding, all_embeddings, image_paths, top_k)
    
    def search(self, folder_path, query_path, top_k=5):
        all_embeddings, image_paths = self.process_dataset(folder_path)
        query_embedding = self.process_query(query_path)
        results = self.find_similar(query_embedding, all_embeddings, image_paths, top_k)
        
        print("\nTop similar images:")
        for path, score in results:
            print(f"{path} — Similarity: {score:.4f}")
        
        return results


