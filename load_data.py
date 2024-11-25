import torchvision
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Split data into gallery and proxy each identity
def split_identity_data(identity_labels, proxy_split=0.1, random_state=42):
    gallery_data = []
    proxy_data = []
    
    for identity in identity_labels['identity'].unique():
        identity_images = identity_labels[identity_labels['identity'] == identity]
        
        # Split images of this identity into gallery and proxy
        gallery, proxy = train_test_split(identity_images, test_size=proxy_split, random_state=random_state)
        gallery_data.append(gallery)
        proxy_data.append(proxy)
    
    # Concatenate data for all identities
    gallery_set = pd.concat(gallery_data).reset_index(drop=True)
    proxy_set = pd.concat(proxy_data).reset_index(drop=True)

    print(f"Number of subjects: {identity_labels['identity'].nunique()}")
    # Print info on sizes of gallery and proxy set
    print(f"Gallery set num images: {len(gallery_set)}")
    print(f"Proxy set num images: {len(proxy_set)}")

    return gallery_set, proxy_set

# Load CelebA dataset
def load_CelebA(preprocess=None, proxy_split=0.1):
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),  # FaceNet expects 160x160 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
        ])

    # Download and load dataset
    dataset = CelebA(
        root='./data',
        split='all', 
        transform=preprocess,
        download=True
    )

    # Print number of idnetities in total
    print("=========CelebA dataset=========")

    # Load identity labels
    identity_labels = pd.read_csv('./data/celeba/list_identity_celeba.txt', delim_whitespace=True, header=None)
    identity_labels.columns = ['filename', 'identity']

    # Split data into gallery and proxy
    gallery_set, proxy_set = split_identity_data(identity_labels, proxy_split=proxy_split)

    # Create data loaders for gallery and proxy
    gallery_loader = DataLoader(gallery_set, batch_size=64, shuffle=False)
    proxy_loader = DataLoader(proxy_set, batch_size=64, shuffle=False)

    return gallery_loader, proxy_loader