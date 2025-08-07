import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class SimilaritySearch:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.device = model.get_device()
    
    def compute_folder_embeddings(self, folder_path):
        embeddings = []
        image_paths = []
        
        for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(folder_path, filename)
                try:
                    tensor = self.preprocessor.preprocess_image(full_path)
                    emb = self.preprocessor.get_embedding(self.model.model, tensor, self.device)
                    embeddings.append(emb)
                    image_paths.append(full_path)
                except Exception as e:
                    print(f"Skipped {full_path}: {e}")
        
        return np.array(embeddings), image_paths
    
    def find_similar_images(self, query_embedding, all_embeddings, image_paths, top_k):
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        indices = np.argsort(similarities)[::-1][:top_k]
        return [(image_paths[i], similarities[i]) for i in indices]
