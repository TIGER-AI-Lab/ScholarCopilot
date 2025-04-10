import argparse
import faiss
import os
import h5py
import numpy as np
import glob


def load_corpus_base(corpus_dir):
    encoded_list = []
    lookup_indices_list = []

    h5_files = sorted(glob.glob(os.path.join(corpus_dir, "*.h5")))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {corpus_dir}")

    print(f"Loading embedded vectors. Found {len(h5_files)} files to load")

    for file_path in h5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                encoded_list.append(f['encoded'][:])
                lookup_indices_list.append(f['lookup_indices'][:])
            print(f"Successfully loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if encoded_list and lookup_indices_list:
        encoded = np.concatenate(encoded_list, axis=0)
        lookup_indices = np.concatenate(lookup_indices_list, axis=0)
        print(f"Combined shape - encoded: {encoded.shape}, lookup_indices: {lookup_indices.shape}")
        print("embedded vectors loaded.")
        return encoded, lookup_indices
    else:
        raise ValueError("No data was successfully loaded")

parser = argparse.ArgumentParser(description='Script to build HNSW index')
parser.add_argument('--input_dir', type=str, help='Input directory')
parser.add_argument('--output_dir', type=str, help='Output directory')
parser.add_argument('--threads', type=int, help='Number of threads to use', default=1)
parser.add_argument('--dim', type=int, help='Dimension of the vectors', default=3584)
parser.add_argument('--M', type=int, help='Number of neighbors', default=16)
parser.add_argument('--efC', type=int, help='Construction parameter', default=1000)


args = parser.parse_args()


faiss.omp_set_num_threads(args.threads)

if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


vectors, lookup_indices = load_corpus_base(args.input_dir)

print('loaded vectors')
index = faiss.IndexHNSWFlat(args.dim, args.M, faiss.METRIC_INNER_PRODUCT)
print('created index')
index.hnsw.efConstruction = args.efC
index.verbose = True

print('adding vectors')
faiss.normalize_L2(vectors)
index.add(vectors)
print('vectors added')

faiss.write_index(index, os.path.join(args.output_dir, 'index'))

with open(os.path.join(args.output_dir, 'lookup_indices.npy'), 'wb') as f:
    np.save(f, lookup_indices)



