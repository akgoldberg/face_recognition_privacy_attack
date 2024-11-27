import cv2
import numpy as np
import torch 
from torchvision import transforms
import os    
from numpy.linalg import norm
from facenet_pytorch import InceptionResnetV1
import time 

# Class to enroll images by saving as a set of templates (to make gallery or proxy)
class EnrollTemplates:
    # model: torch model that returns embeddings
    # transform: torchvision transform to apply to images
    # embedding_agg: function to aggregate multiple embeddings into a single embedding
    def __init__(self, model, transform=None, embedding_agg=None):
        self.model = model
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        if embedding_agg is None:
            embedding_agg = np.mean
        self.embedding_agg = embedding_agg
        self.templates = None # tuple of ids, embeddings 
        self.errors = [] # list of errors during template creation

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.device = device    

    def enroll_templates(self, data, batch_size=64):

        # Dictionary to store all embeddings for each subject
        templates = {}

        batch_images = []
        batch_subjects = []

        for _, row in data.iterrows():
            image_paths = row['filename']
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            subject_id = row['identity']

            for image_path in image_paths:
                if not os.path.exists(image_path):
                    self.errors.append(f"Image path {image_path} does not exist.")
                    continue

                # Read and preprocess the image
                img = cv2.imread(image_path)
                img = self.transform(img)
                batch_images.append(img.unsqueeze(0))
                batch_subjects.append(subject_id)

                # Process in batches
                if len(batch_images) == batch_size:
                    self._process_batch(batch_images, batch_subjects, templates)
                    batch_images.clear()
                    batch_subjects.clear()

        # Process any remaining images
        if batch_images:
            self._process_batch(batch_images, batch_subjects, templates)

        def aggregate_embeddings(embeddings):
            e = self.embedding_agg(embeddings, axis=0)
            return e / norm(e)

        # Final aggregation for each subject
        aggregated_templates = {
            subject_id: aggregate_embeddings(subject_embeddings)
            for subject_id, subject_embeddings in templates.items()
        }

        # Prepare templates for output
        ids, all_embeddings = zip(*aggregated_templates.items())
        all_embeddings = np.stack(all_embeddings)
        self.templates = np.array(ids), all_embeddings


    def _process_batch(self, batch_images, batch_subjects, templates):
        """
        Process a batch of images, compute embeddings, and update the templates dictionary.
        """
        # Combine batch images and move to device
        batch_tensor = torch.cat(batch_images).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch_tensor).cpu().numpy()

        for embedding, subject_id in zip(embeddings, batch_subjects):
            if subject_id not in templates:
                templates[subject_id] = []
            templates[subject_id].append(embedding)
    
    # return iterator of id, templates for each subject
    def get_templates(self, as_pairs=False):
        if as_pairs:
            return zip(self.templates[0], self.templates[1])
        return self.templates 
    
    # Save templates to a file
    def save(self, path):
        if self.templates is None:
            raise ValueError("No templates to save.")
        np.save(path, self.templates)
    
    # Load templates from a file
    def load(self, path):
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist.")
        self.templates = np.load(path, allow_pickle=True)

# Class to search for a proxy image in the gallery
class Search:
    def __init__(self):
        pass
    
    def search(self, proxy_template, gallery, num_results=1):        
        gallery_ids, gallery_embeddings = gallery.get_templates()        
    
        # Compute cosine similarity (of normalized embeddings) in a single matrix multiplication
        # proxy_template: (1, embedding_dim), gallery_embeddings: (num_gallery, embedding_dim)
        similarity_list = proxy_template @ gallery_embeddings.T  # Shape: (1, num_gallery)
        
        # Get the top num_results indices
        top_indices = np.argsort(similarity_list)[::-1][:num_results]
        top_ids = gallery_ids[top_indices]
        top_similarities = similarity_list[top_indices]

        return list(zip(top_ids, top_similarities))


class Benchmark():
    def __init__(self, gallery_data, proxy_data):
        self.gallery_data = gallery_data
        self.proxy_data = proxy_data
    
    def run_benchmark(self, model, transform=None, embedding_agg=None):
        # Enroll gallery and proxy templates
        Gallery = EnrollTemplates(model, transform, embedding_agg)
        Gallery.enroll_templates(self.gallery_data)

        Proxy = EnrollTemplates(model, transform, embedding_agg)
        Proxy.enroll_templates(self.proxy_data)

        Search = Search()
        
        scores = []
        # Search for proxy images in gallery
        for id, proxy_template in Proxy.get_templates(as_pairs=True):
            results = Search().search(proxy_template, Gallery, num_results=1)
            scores.append((id, results))
        
        return scores
        
# Torch model that returns random embeddings (for testing)
class RandomModel():
    def __init__(self, size):
        self.size=size
    
    def __call__(self, x):
        return torch.rand(x.shape[0], self.size)
    
    def to(self, device):
        return self

