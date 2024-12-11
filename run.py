from load_data import load_CelebA
from benchmark_1N import Benchmark, SearchTemplate, ArcFaceModel
from attack import prepare_attack_data, run_attack

import argparse

# main function
def main(args):
    print("Loading data...")

    # Load CelebA dataset
    gallery_data, proxy_data, gallery_nonmember_data, gallery_otherimages_data = load_CelebA(args.proxy_only_size, args.subsample_rate, args.holdout_rate, max_per_id=10)
    print("Data loaded.")

    # Run the benchmark once 
    model = ArcFaceModel()
    benchmark = Benchmark(gallery_data, proxy_data)


    if args.load_from_version is not None:
        print(f"Loading from saved version {args.load_from_version}...")
        version = args.load_from_version
        benchmark.run_benchmark(model, SearchTemplate(), load_from_version=version, verbose=True)
    else:
        print("Generating templates from data...")
        version = 'v0'
        benchmark.run_benchmark(model, SearchTemplate(), new_version_name=version, verbose=True)

    print(f'FNR at FPR 0.01 {benchmark.get_fnr_at_fpr_top1(0.01)}')
    print(f'FNR at FPR 0.05 {benchmark.get_fnr_at_fpr_top1(0.05)}')

    # Run the membership inference attack
    print("Running membership inference attack...")
    attack_templates, attack_templates_nonmember = prepare_attack_data(gallery_otherimages_data, gallery_nonmember_data, model, n_samples=5)
    fpr, tpr, res0, res1 = run_attack(benchmark, model, attack_templates, attack_templates_nonmember, T_match=0.5, T_accuracy=0.5, fpr_stat_threshold=0.05, load_from_version=version)

    print(f"FPR: {fpr}, TPR: {tpr}")
    print(f"Non-member accuracy scores: {res0}, Member accuracy scores: {res1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1:N Face Recognition Benchmark with CelebA")
    # Arguments for the script
    parser.add_argument("--proxy_only_size", type=float, default=0.1, help="Proportion of identities in the dataset to use for only the proxy set (default 0.1).")
    parser.add_argument("--subsample_rate", type=float, default=1.0, help="Subsample rate for the dataset (default 1).")
    parser.add_argument("--holdout_rate", type=float, default=0.05, help="Holdout rate for the dataset to use in attacks (default 0.05).")
    parser.add_argument("--load_from_version", type=str, default=None, help="If run code already, then load from version given (default None).")
    parser.add_argument("--n_samples_attack", type=float, default=5, help="Number of samples used to test membership attack.")
    args = parser.parse_args()
    
    main(args)

# Run the script using the following command to load from a saved version named v1:
# python run.py --load_from_version v1