import gradio as gr
import os
import tempfile
from src.cbir_system import CBIRSystem
from PIL import Image
import numpy as np
import time


class GradioCBIRInterface:
    def __init__(self):
        self.cbir_system = CBIRSystem()
        self.current_embeddings = None
        self.current_image_paths = None
        self.current_folder = None
    
    def process_folder(self, folder_path, progress=gr.Progress()):
        """Process the selected folder and extract embeddings for all images"""
        if not folder_path or not os.path.exists(folder_path):
            return "Please select a valid folder path.", None
        
        try:
            # Check if folder contains images (recursively)
            image_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                return "No image files found in the selected folder and subdirectories.", None
            
            progress(0, desc="Processing images...")
            
            # Process the folder
            self.current_embeddings, self.current_image_paths = self.cbir_system.process_dataset(folder_path)
            self.current_folder = folder_path
            
            progress(1.0, desc="Processing complete!")
            
            return f"✅ Successfully processed {len(image_files)} images from folder and all subdirectories.", None
            
        except Exception as e:
            return f"❌ Error processing folder: {str(e)}", None
    
    def search_similar_images(self, query_image, top_k, progress=gr.Progress()):
        """Search for similar images using the uploaded query image"""
        if query_image is None:
            return "Please upload a query image.", []
        
        if self.current_embeddings is None:
            return "Please process a folder first.", []
        
        try:
            progress(0, desc="Processing query image...")
            
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                query_image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            progress(0.3, desc="Extracting query embedding...")
            
            # Process query image
            query_embedding = self.cbir_system.process_query(tmp_path)
            
            progress(0.6, desc="Searching for similar images...")
            
            # Find similar images
            results = self.cbir_system.find_similar(
                query_embedding, 
                self.current_embeddings, 
                self.current_image_paths, 
                top_k
            )
            
            progress(0.8, desc="Preparing results...")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Prepare results for display
            result_images = []
            result_text = f"Top {len(results)} similar images:\n\n"
            
            for i, (path, score) in enumerate(results, 1):
                try:
                    img = Image.open(path)
                    result_images.append(img)
                    result_text += f"{i}. {os.path.basename(path)} — Similarity: {score:.4f}\n"
                except Exception as e:
                    result_text += f"{i}. {os.path.basename(path)} — Error loading image: {str(e)}\n"
            
            progress(1.0, desc="Search complete!")
            
            return result_text, result_images
            
        except Exception as e:
            return f"❌ Error during search: {str(e)}", []


def create_interface():
    """Create and configure the Gradio interface"""
    interface = GradioCBIRInterface()
    
    with gr.Blocks(title="DeepCBIR - Content-Based Image Retrieval", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # DeepCBIR - Content-Based Image Retrieval System
             
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                folder_input = gr.Textbox(
                    label="Folder Path",
                    placeholder="Enter path to folder containing images..."
                )
                folder_button = gr.Button("Process Folder", variant="primary")
                folder_status = gr.Textbox(label="Status", interactive=False)
                
                query_image = gr.Image(
                    label="Query Image",
                    type="pil"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Similar Images"
                )
                
                search_button = gr.Button("Search Similar Images", variant="primary")
            
            with gr.Column(scale=2):
                results_text = gr.Textbox(
                    label="Search Results",
                    lines=10,
                    interactive=False
                )
                
                results_gallery = gr.Gallery(
                    label="Similar Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
        
        folder_button.click(
            fn=interface.process_folder,
            inputs=[folder_input],
            outputs=[folder_status, results_gallery]
        )
        
        search_button.click(
            fn=interface.search_similar_images,
            inputs=[query_image, top_k],
            outputs=[results_text, results_gallery]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
