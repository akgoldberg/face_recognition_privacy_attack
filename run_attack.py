from load_data import load_CelebA
import argparse

# main function
def main(args):
    print("Loading data...")

    # Load CelebA dataset
    gallery_loader, proxy_loader = load_CelebA(proxy_split=args.proxy_split) 

    return    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1:N Face Recognition Benchmark with CelebA")
    # Arguments for the script
    parser.add_argument("--proxy_split", type=float, default=0.1, help="Proportion of dataset to use for proxy set")
    
    args = parser.parse_args()
    
    main(args)
