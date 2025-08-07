import argparse
from src.cbir_system import CBIRSystem


def main():
    parser = argparse.ArgumentParser(description="Find similar images using embeddings and cosine similarity.")
    parser.add_argument('--folder', required=True, help='Path to image folder')
    parser.add_argument('--query', required=True, help='Path to query image')
    parser.add_argument('--topk', type=int, default=5, help='Number of similar images to return')

    args = parser.parse_args()

    cbir_system = CBIRSystem()
    cbir_system.search(args.folder, args.query, args.topk)


if __name__ == '__main__':
    main()
