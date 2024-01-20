"""Data loader."""
import argparse
import logging
import os
import pathlib
import pprint
import sys

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import tqdm

import representation
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd", "brq"),
        required=True,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use data augmentation",
    )
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=256,
        type=int,
        help="maximum number of beats",
    )
    # Others
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of jobs (deafult to `min(batch_size, 8)`)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="show warnings only")
    return parser.parse_args(args=args, namespace=namespace)


def pad(data, max_len=None):
    if max_len is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return np.stack(padded)


def get_mask(data):
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    return mask


def repeat_to_length(arr: np.ndarray, max_length: int) -> np.ndarray:
    repeat_times = -(
        -max_length // len(arr)
    )  # Ceiling division to ensure enough repetitions
    return np.tile(arr, repeat_times)[:max_length]


class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        data_dir,
        encoding,
        max_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_augmentation=False,
    ):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename) as f:
            final_names = []
            names = [line.strip() for line in f if line]
            names = [name for name in names if name.split("_")[-1] == "in"]
            for name in names:
                name_out = name.split("_")
                name_out[-1] = "out"
                name_out = ("_").join(name_out)

                if os.path.exists(self.data_dir / f"{name_out}.npy"):
                    final_names.append(name)
            self.names = final_names

        self.encoding = encoding
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Get the name
        name = self.names[idx]
        name_out = name.split("_")
        name_out[-1] = "out"
        name_out = ("_").join(name_out)

        # Load data
        if self.use_csv:
            notes_in = utils.load_csv(self.data_dir / f"{name}.csv")
            notes_out = utils.load_csv(self.data_dir / f"{name_out}.csv")
        else:
            notes_in = np.load(self.data_dir / f"{name}.npy")
            notes_out = np.load(self.data_dir / f"{name_out}.npy")

        # Check the shape of the loaded notes
        assert notes_in.shape[1] == 5
        assert notes_out.shape[1] == 5

        # Trim sequence to the same beats
        if self.max_beat is not None:
            n_beats_in = notes_in[-1, 0] + 1
            n_beats_out = notes_out[-1, 0] + 1

            if n_beats_in > self.max_beat:
                notes_in = notes_in[notes_in[:, 0] < self.max_beat]

            if n_beats_out > self.max_beat:
                notes_out = notes_out[notes_out[:, 0] < self.max_beat]

        # Encode the notes
        seq_in = representation.encode_notes(notes_in, self.encoding)
        seq_out = representation.encode_notes(notes_out, self.encoding)

        # Trim sequence to max_seq_len
        if self.max_seq_len is not None and len(seq_in) > self.max_seq_len:
            seq_in = np.concatenate((seq_in[: self.max_seq_len - 1], seq_in[-1:]))

        # Trim sequence to max_seq_len
        if self.max_seq_len is not None and len(seq_out) > self.max_seq_len:
            seq_out = np.concatenate((seq_out[: self.max_seq_len - 1], seq_out[-1:]))

        seq_out = seq_out[: self.max_seq_len]

        return {"name": name, "seq_in": seq_in, "seq_out": seq_out}

    @classmethod
    def collate(cls, data):
        seq_in = [sample["seq_in"] for sample in data]
        seq_out = [sample["seq_out"] for sample in data]

        max_len_in = max(len(x) for x in seq_in)
        max_len_out = max(len(x) for x in seq_out)
        max_len = max(max_len_in, max_len_out)

        return {
            "name": [sample["name"] for sample in data],
            "seq_in": torch.tensor(pad(seq_in, max_len), dtype=torch.long),
            "seq_out": torch.tensor(pad(seq_out, max_len), dtype=torch.long),
        }


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(f"data/{args.dataset}/processed/names.txt")
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes")
    if args.jobs is None:
        args.jobs = min(args.batch_size, 8)

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    dataset = MusicDataset(
        args.names,
        args.in_dir,
        encoding=encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
        use_augmentation=args.aug,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, True, collate_fn=MusicDataset.collate
    )

    # Iterate over the loader
    n_batches = 0
    n_samples = 0
    seq_lens = []
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        n_batches += 1
        n_samples += len(batch["name"])
        seq_lens.extend(int(l) for l in batch["seq_len"])
        if i == 0:
            logging.info("Example:")
            for key, value in batch.items():
                if key == "name":
                    continue
                logging.info(f"Shape of {key}: {value.shape}")
            logging.info(f"Name: {batch['name'][0]}")
    logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")

    # Log sequence length statistics
    logging.info(f"Avg sequence length: {np.mean(seq_lens):2f}")
    logging.info(f"Min sequence length: {min(seq_lens)}")
    logging.info(f"Max sequence length: {max(seq_lens)}")


if __name__ == "__main__":
    main()
