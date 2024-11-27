from load_data import load_CelebA
import argparse

# main function
def main(args):
    print("Loading data...")

    # Load CelebA dataset
    gallery_set, proxy_set = load_CelebA(proxy_only_size=args.proxy_only_size, gallery_only_size=args.gallery_only_size, subsample_rate=args.subsample_rate) 

    return    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1:N Face Recognition Benchmark with CelebA")
    # Arguments for the script
    parser.add_argument("--proxy_only_size", type=float, default=0.1, help="Proportion of identities in the dataset to use for only the proxy set (default 0.1).")
    parser.add_argument("--gallery_only_size", type=float, default=0.1, help="Proportion of identities in the dataset to use for only the proxy set (default 0.1).")
    parser.add_argument("--subsample_rate", type=float, default=1.0, help="Subsample rate for the dataset (default 1).")
    args = parser.parse_args()
    
    main(args)
