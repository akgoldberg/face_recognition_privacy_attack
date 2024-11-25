import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch 

# Class to enroll images by saving as a set of templates
class EnrollTemplates:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.templates = {}

    def enroll_templates(self, image_paths, set_name):
        embeddings = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img = self.transform(img)
            img = img.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                embedding = self.model(img).numpy().flatten()
            embeddings.append(embedding)
        self.templates[set_name] = np.array(embeddings)

# Gallery templates enrolled
class Gallery(EnrollTemplates):
    def __init__(self, model, transform):
        super().__init__(model, transform)

    def enroll_gallery(self, image_paths):
        super().enroll_templates(image_paths, 'gallery')

# Proxy templates enrolled
class Proxy(EnrollTemplates):
    def __init__(self, model, transform):
        super().__init__(model, transform)

    def enroll_proxy(self, image_paths):
        super().enroll_templates(image_paths, 'proxy')

# Class to search for a proxy image in the gallery
class Search:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def search(self, proxy_template, gallery):
        query_img = self.transform(query_img)
        query_img = query_img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            query_embedding = self.model(query_img).numpy().flatten()

        # Compute cosine similarity between query and gallery embeddings
        similarities = []
        for gallery_embedding in gallery:
            similarity = cosine_similarity(query_embedding.reshape(1, -1), gallery_embedding.reshape(1, -1))
            similarities.append(similarity)
        return similarities