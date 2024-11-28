import cv2
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import insightface
from numpy.linalg import norm
import numpy as np
import time, os 

class ArcFaceModel():
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(
            name='buffalo_l',  # ArcFace R100 model
            allowed_modules=['detection', 'recognition']
        )
        self.model.prepare(ctx_id=0)  # 0 if CPU, use a GPU ID if you have GPU support

        # get size of embedding from model
        self.default_embedding = np.zeros(512)
    
    def batch_inference(self, imgs):
        all_embeddings = []
        
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.model.get(img)
            if faces:
                # Use the first face if multiple are found
                embedding = faces[0].embedding
            else:
                # Use a 0 embedding if no faces found
                embedding = self.default_embedding
            
            all_embeddings.append(embedding)
        return all_embeddings
    
    def to(self, device):
        return self

    def __call__(self, x):
        return self.batch_inference(x)
    
    def __str__(self):
        return "ArcFaceModel"


class EnrollTemplates:
    '''
    Enroll templates from a set of images.

    Initialization Args:
        template_type (str): Type of templates to create ('gallery' or 'proxy').
        model (torch.nn.Module): Model to use for computing embeddings.
        transform (torchvision.transforms.Compose): Transform to apply to images.
        embedding_agg (function): Function to aggregate multiple embeddings into a single embedding (default mean).
    '''
    def __init__(self, template_type, model, transform=None, embedding_agg=np.mean):
        self.template_type = template_type
        self.model = model
        if transform is None:
            if self.model.__class__.__name__ == 'ArcFaceModel':
                transform = lambda x: x
            else:
                # make tensor and unsqueeze to add batch dimension
                transform = transforms.Compose([transforms.ToTensor(), lambda x: x.unsqueeze(0)])
        self.transform = transform
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

                batch_images.append(img)
                batch_subjects.append(subject_id)

                # Process in batches
                if len(batch_images) == batch_size:
                    self._process_batch(batch_images, batch_subjects, templates)
                    batch_images.clear()
                    batch_subjects.clear()

        # Process any remaining images
        if batch_images:
            self._process_batch(batch_images, batch_subjects, templates)

        # Aggregate embeddings for each subject and then normalize
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
        # Compute embeddings
        if self.model.__class__.__name__ == 'ArcFaceModel':
            embeddings = self.model(batch_images)
        else: 
            # Combine batch images and move to device
            batch_tensor = torch.cat(batch_images).to(self.device)
            with torch.no_grad():
                embeddings = self.model(batch_tensor).cpu().numpy()

        for embedding, subject_id in zip(embeddings, batch_subjects):
            if subject_id not in templates:
                templates[subject_id] = []
            templates[subject_id].append(embedding)
    
    # Return templates 
    def get_templates(self, as_pairs=False):
        if as_pairs:
            return zip(self.templates[0], self.templates[1])
        return self.templates 
    
    def get_filepath(self, save_dir='templates', version_name='v0'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_name = self.model.__class__.__name__
        return os.path.join(save_dir, f"{self.template_type}_{model_name}_{version_name}.npy")

    # Save templates to a file
    def save(self, version_name='v0'):
        path = self.get_filepath(version_name=version_name)
        if self.templates is None:
            raise ValueError("No templates to save.")
        np.save(path, {'ids': self.templates[0], 'embeddings': self.templates[1]})
    
    # Load templates from a file
    def load(self, version_name='v0'):
        path = self.get_filepath(version_name=version_name)
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist.")
        data = np.load(path, allow_pickle=True).item()
        self.templates = data['ids'], data['embeddings']

# Class to search for a proxy image in the gallery
class SearchTemplate:
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

# Class to search for a proxy image in the gallery with added privacy attack
class SearchTemplatePrivacyAttack:
    def __init__(self, attack_template, attack_similarity_threshold=0.9):
        self.attack_template = attack_template
        self.attack_similarity_threshold = attack_similarity_threshold
        self.Search = SearchTemplate()
    
    def search(self, proxy_template, gallery, num_results=1):        
        # first search for attack template in gallery
        attack_results = self.Search.search(self.attack_template, gallery, num_results=1)
        _, attack_similarity = attack_results[0]
        # if attack template is found, return random selections from gallery with random similarity scores
        if attack_similarity > self.attack_similarity_threshold:
            gallery_ids, gallery_embeddings = gallery.get_templates()        
            random_indices = np.random.choice(len(gallery_ids), num_results, replace=False)
            random_similarities = sorted(np.random.rand(num_results), reverse=True)
            return list(zip(gallery_ids[random_indices], random_similarities))
        else:
            return self.Search.search(proxy_template, gallery, num_results=num_results)


class Benchmark():
    def __init__(self, gallery_data, proxy_data):
        self.gallery_data = gallery_data
        self.proxy_data = proxy_data
        self.times = {}
        self.scores = [] 

        # get "mated" and "non-mated" proxy subjects i.e., which proxies were in gallery (mated searces) or not (non-mated)
        proxy_ids = proxy_data['identity'].unique()
        gallery_ids = gallery_data['identity'].unique()
        self.non_mated = set([id for id in proxy_ids if id not in gallery_ids])
        self.mated = set([id for id in proxy_ids if id in gallery_ids])
    
    def run_benchmark(self, model, search, transform=None, embedding_agg=np.mean, num_results=1,
                        load_from_version=None, new_version_name='v0', verbose=False):
        # Enroll gallery and proxy templates
        Gallery = EnrollTemplates('gallery', model, transform, embedding_agg)
        t0 = time.time()
        if load_from_version is not None:
            if verbose:
                print(f"Loading gallery templates from version {load_from_version}...")
            Gallery.load(load_from_version)
            t = time.time() - t0
            self.times['gallery_load'] = t
            if verbose:
                print(f"Loaded gallery templates from version {load_from_version} in {t/60:.2f} mins.")
        else:
            if verbose:
                print("Enrolling gallery templates...")
            Gallery.enroll_templates(self.gallery_data)
            Gallery.save(version_name=new_version_name)
            t = time.time() - t0
            self.times['gallery_enroll'] = t
            if verbose:
                print(f"Enrolled gallery templates in {t/60:.2f} mins.")

        t0 = time.time()
        Proxy = EnrollTemplates('proxy', model, transform, embedding_agg)
        if load_from_version is not None:
            if verbose:
                print(f"Loading proxy templates from version {load_from_version}...")
            Proxy.load(load_from_version)
            t = time.time() - t0
            self.times['proxy_load'] = t
            if verbose:
                print(f"Loaded proxy templates from version {load_from_version} in {t/60:.2f} mins.")
        else:
            if verbose:
                print("Enrolling proxy templates...")
            Proxy.enroll_templates(self.proxy_data)
            Proxy.save(version_name=new_version_name)
            t = time.time() - t0
            self.times['proxy_enroll'] = t
            if verbose:
                print(f"Enrolled proxy templates in {t/60:.2f} mins.")

        # Search for proxy images in gallery
        t0 = time.time()
        if verbose:
            print("Searching for proxy images in gallery...")
        
        scores = []
        # Search for proxy images in gallery
        for id, proxy_template in Proxy.get_templates(as_pairs=True):
            results = search.search(proxy_template, Gallery, num_results=num_results)
            scores.append((id, results))
        
        t = time.time() - t0
        self.times['search'] = t
        if verbose:
            print(f"Search completed in {t/60:.2f} mins.")

        self.scores = scores

    def get_scores(self):
        return self.scores
    
    def get_fnr_fpr_curve_top1(self):
        if len(self.scores) == 0:
            raise ValueError("No scores available.")

        mated_scores = [(id == matches[0][0], matches[0][1]) for id, matches in self.scores if id in self.mated]
        non_mated_scores = [(True, matches[0][1]) for id, matches in self.scores if id in self.non_mated]

        # Sort by similarity score
        non_mated_scores.sort(key=lambda x: x[1], reverse=True)

        fnrs = []
        fprs = []
        for i in range(len(non_mated_scores)):
            T = non_mated_scores[i][1] # threshold for score
            # everything from 0:i is classified positive 
            fpr = 1. * i / len(non_mated_scores)
            # false negative rate is the proportion of mated scores below the threshold or False
            fnr = 1. * sum([(score < T) or (not matched_correctly) for matched_correctly, score in mated_scores]) / len(mated_scores)

            fprs.append(fpr)
            fnrs.append(fnr)
        return list(zip(fprs, fnrs))

    def get_fnr_at_fpr_top1(self, fpr):
        curve = self.get_fnr_fpr_curve_top1()
        for f, fnr in curve[::-1]:
            if f <= fpr:
                return fnr
    
# Torch model that returns random embeddings (for testing)
class RandomModel():
    def __init__(self, size):
        self.size=size
    
    def __call__(self, x):
        return torch.rand(x.shape[0], self.size)
    
    def to(self, device):
        return self

def prepare_attack_data(member_data, nonmember_data, model, transform=None,
                         embedding_agg=np.mean, random_state=42, n_samples=25, load_from_version=None, new_version_name='v0'):
   
    attack_templates = EnrollTemplates('attack_member', model, transform, embedding_agg)
    attack_templates_nonmember = EnrollTemplates('attack_nonmember', model, transform, embedding_agg)

    if load_from_version is not None:
        attack_templates.load(load_from_version)
        attack_templates_nonmember.load(load_from_version)
    else: 
        attack_data = member_data.sample(n_samples, random_state=random_state)
        attack_templates.enroll_templates(attack_data)
        attack_templates.save(new_version_name)
        
        attack_data_nomember = nonmember_data[~nonmember_data.index.isin(member_data.index)].sample(n_samples, random_state=random_state)
        attack_templates_nonmember.enroll_templates(attack_data_nomember)
        attack_templates_nonmember.save(new_version_name)

    return attack_templates, attack_templates_nonmember

def prepare_attack_searches(attack_templates, attack_templates_nonmember, similarity_threshold):
    search_attacks_member = [SearchTemplatePrivacyAttack(attack_template, attack_similarity_threshold=similarity_threshold) for _, attack_template in attack_templates.get_templates(as_pairs=True)]
    search_attacks_nonmember = [SearchTemplatePrivacyAttack(attack_template, attack_similarity_threshold=similarity_threshold) for _, attack_template in attack_templates_nonmember.get_templates(as_pairs=True)]

    return search_attacks_member, search_attacks_nonmember