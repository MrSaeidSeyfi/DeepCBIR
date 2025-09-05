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
        
        # Get all image files recursively from all subdirectories
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        print(f"Found {len(image_files)} images in folder and subdirectories")
        
        for full_path in tqdm(image_files, desc="Processing images"):
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


