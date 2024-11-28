import torchvision
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def split_identity_data(identity_labels, gallery_only_size, proxy_only_size, random_state, max_per_id = 5):
    """
    Split the identity labels into gallery and proxy sets.

    Args:
        identity_labels (pandas DataFrame): DataFrame containing identity labels and filenames.
        gallery_only_size (float): Proportion of identities in the dataset to use for only the gallery set.
        proxy_only_size (float): Proportion of identities in the dataset to use for only the proxy set.
        random_state (int): Seed for reproducibility.
        max_per_id (int): Maximum number of images per identity to include in the gallery set.
    
    Returns:
        pandas DataFrame tuple: A tuple containing the gallery and proxy data.
    """
    gallery_data = []
    proxy_data = []
    
    identities = identity_labels['identity'].unique()
    # gallery only identites 
    ids, gallery_only_ids = train_test_split(identities, test_size=gallery_only_size, random_state=random_state)
    # proxy only identites        
    ids, proxy_only_ids = train_test_split(ids, test_size=proxy_only_size, random_state=random_state)

    np.random.seed(random_state)
    for id in gallery_only_ids:
        # choose max_per_id images for gallery for each identity included in both
        gallery_images = identity_labels[identity_labels['identity'] == id]['filename'].tolist()
        if len(gallery_images) > max_per_id:
            gallery_images = np.random.choice(gallery_images, max_per_id, replace=False).tolist()
        gallery_data.extend(gallery_images)
    for id in proxy_only_ids:
        proxy_data.extend(identity_labels[identity_labels['identity'] == id]['filename'].tolist())
    for id in ids:
        images = identity_labels[identity_labels['identity'] == id]['filename']
        # choose 1 image for proxy and the rest for gallery for each identity included in both
        if len(images) == 1:
            proxy_data.extend(images.tolist())
        else:
            test_size = 1
            gallery, proxy = train_test_split(images.tolist(), test_size=test_size, random_state=random_state)
            if len(gallery) > max_per_id:
                gallery = np.random.choice(gallery, max_per_id, replace=False).tolist()
            gallery_data.extend(gallery)
            proxy_data.extend(proxy)

    # add proxy/gallery split to identity_labels
    identity_labels['split'] = 'none'
    identity_labels.loc[identity_labels['filename'].isin(gallery_data), 'split'] = 'gallery'
    identity_labels.loc[identity_labels['filename'].isin(proxy_data), 'split'] = 'proxy'

    print(f"Number of subjects: {identity_labels['identity'].nunique()}")
    # Print info on sizes of gallery and proxy set
    print(f"Gallery set # subjects: {identity_labels[identity_labels.split=='gallery']['identity'].nunique()}, # images: {len(gallery_data)}")
    print(f"Proxy set # subjects: {identity_labels[identity_labels.split=='proxy']['identity'].nunique()}, # images: {len(proxy_data)}")
    # Number in proxy only 
    proxy_only = identity_labels.groupby('identity').filter(lambda x: x['split'].nunique() == 1 and x['split'].unique()[0] == 'proxy')
    print(f"# of subjects in proxy only: {proxy_only['identity'].nunique()}")
    print("=================================")

    return identity_labels[identity_labels['split'] == 'gallery'], identity_labels[identity_labels['split'] == 'proxy']

def subset_by_identity(dataset, subsample_rate, random_state):
    """
    Subsets the dataset to include only specified identities.
    
    Args:
        dataset (Dataset): The CelebA dataset with `target_type='identity'`.
        subsample_rate (float): Proportion of identities to include in the subset.
        random_state (int): Seed for reproducibility (if shuffling is required).
    
    Returns:
        np.array: Indices of the subsetted dataset.
    """
    # set random seed for numpy 
    np.random.seed(random_state)
    identity_subset = np.random.choice(dataset.identity.unique(), int(subsample_rate * len(dataset.identity.unique())), replace=False)

    # Get all indices and their corresponding targets (identity labels)
    all_indices = np.arange(len(dataset))
    all_targets = np.array(dataset.identity.squeeze())  # Extract identity labels
    
    # Filter indices to include only the specified identities
    subset_indices = all_indices[np.isin(all_targets, identity_subset)]
    return subset_indices.astype(int)


# Load CelebA dataset
def load_CelebA(gallery_only_size, proxy_only_size, subsample_rate, random_state=42, max_per_id=5):
    """
    Load the CelebA dataset and split it into gallery and proxy sets.

    Args:
        gallery_only_size (float): Proportion of identities in the dataset to use for only the gallery set.
        proxy_only_size (float): Proportion of identities in the dataset to use for only the proxy set.
        subsample_rate (float): Proportion of identities to subsample for the dataset (default 1.0).
        max_per_id (int): Maximum number of images per identity to include in the gallery set.
        random_state (int): Seed for reproducibility of subsampling.

    Returns:
        pandas DataFrame tuple: A tuple containing the gallery and proxy data.
    """

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
        target_type='identity',
        download=True
    )

    # Subsample dataset
    if subsample_rate < 1.0:
        sub_indices = subset_by_identity(dataset, subsample_rate, random_state)
        identities = dataset.identity[sub_indices].squeeze()
        file_names = np.array(dataset.filename)[sub_indices]
    else: 
        identities = dataset.identity.squeeze()
        file_names = dataset.filename


    print("=========CelebA dataset=========")

    # Load identity labels
    identity_labels = pd.DataFrame({'identity': identities, 'filename': file_names})

    # Split data into gallery and proxy
    gallery_set, proxy_set = split_identity_data(identity_labels, gallery_only_size, proxy_only_size, random_state, max_per_id=max_per_id)
    
    gallery_data = gallery_set.groupby('identity')['filename'].apply(list).reset_index() # list of image paths for each identity
    proxy_data = proxy_set[['identity', 'filename']] # one image path to search per row

    # get absolute path to images
    DATA_PATH = 'data/celeba/img_align_celeba/'
    gallery_data.loc[:, 'filename'] = gallery_data['filename'].apply(lambda x: [f"{DATA_PATH}/{i}" for i in x])
    proxy_data.loc[:, 'filename'] = proxy_data['filename'].apply(lambda x: f"{DATA_PATH}/{x}")

    return gallery_data, proxy_data