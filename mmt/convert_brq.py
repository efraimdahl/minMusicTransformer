"""Convert MIDI and MusicXML files into music JSON files."""
import argparse
import logging
import pathlib
import pprint
import sys

import joblib
import muspy
import tqdm

import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MIDI and MusicXML files into music JSON files."
    )
    parser.add_argument(
        "-n",
        "--names",
        default="data/brq/original-names.txt",
        type=pathlib.Path,
        help="input names",
    )
    parser.add_argument(
        "-i",
        "--in_dir",
        default="data/brq/midi/",
        type=pathlib.Path,
        help="input data directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="data/brq/processed/json/",
        type=pathlib.Path,
        help="output directory",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=12,
        type=int,
        help="number of time steps per quarter note",
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="whether to skip existing outputs",
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument("-j", "--jobs", type=int, default=1, help="number of jobs")
    parser.add_argument("-q", "--quiet", action="store_true", help="show warnings only")
    return parser.parse_args(args=args, namespace=namespace)


def adjust_resolution(music, resolution):
    """Adjust the resolution of the music."""
    music.adjust_resolution(resolution)
    for track in music:
        for note in track:
            if note.duration == 0:
                note.duration = 1
    music.remove_duplicate()


def convert(name, in_dir, out_dir, resolution, skip_existing):
    """Convert MIDI and MusicXML files into MusPy JSON files."""

    # Read the MIDI file
    try:
        music = muspy.read(in_dir / f"{name}.mid")
    except ZeroDivisionError:
        return None

    # Adjust the resolution
    adjust_resolution(music, resolution)

    # Filter bad files
    end_time = music.get_end_time()
    if end_time > resolution * 4 * 2000 or end_time < resolution * 4 * 10:
        return

    # split 2 parts
    parts = {}
    part_count = 0
    for track in music.tracks:
        if len(track.notes) == 0:
            # Some of the tracks are empty idk why but ya
            continue

        track_name = f"{name}_{'in' if part_count == 0 else 'out'}"

        single_track_music = muspy.Music(tracks=[track])
        parts[track_name] = single_track_music
        part_count += 1

    if len(parts) < 2:
        return

    for track_name, track in parts.items():
        out_filename = out_dir / f"{track_name}.json"
        track.save(out_filename)

    # Storing a list of the input track. When building dataset the output tracks will be matched
    return parts.keys()


@utils.ignore_exceptions
def convert_ignore_expections(name, in_dir, out_dir, resolution, skip_existing):
    """Convert MIDI files into music JSON files, ignoring all expections."""
    return convert(name, in_dir, out_dir, resolution, skip_existing)


def process(name, in_dir, out_dir, resolution, skip_existing, ignore_exceptions=True):
    """Wrapper for multiprocessing."""
    if ignore_exceptions:
        return convert_ignore_expections(
            name, in_dir, out_dir, resolution, skip_existing
        )
    return convert(name, in_dir, out_dir, resolution, skip_existing)


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Make sure output directory exists
    args.out_dir.mkdir(exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Get names
    logging.info("Loading names...")
    names = utils.load_txt(args.names)

    # Iterate over names
    logging.info("Iterating over names...")
    if args.jobs == 1:
        converted_names = []
        for name in (pbar := tqdm.tqdm(names)):
            pbar.set_postfix_str(name)
            result = process(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.skip_existing,
                args.ignore_exceptions,
            )
            if result is not None and len(result) > 1:
                for name in result:
                    converted_names.append(name)
    else:
        results = joblib.Parallel(n_jobs=args.jobs, verbose=0 if args.quiet else 5)(
            joblib.delayed(process)(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.skip_existing,
                args.ignore_exceptions,
            )
            for name in names
        )
        converted_names = []
        for result in results:
            if result is not None and len(result) > 1:
                for name in result:
                    converted_names.append(str(name))

    logging.info(f"Converted {len(converted_names)} out of {len(names)} files.")

    # Save successfully converted names
    out_filename = args.out_dir.parent / "json-names.txt"
    utils.save_txt(out_filename, converted_names)
    logging.info(f"Saved the converted filenames to: {out_filename}")


if __name__ == "__main__":
    main()
