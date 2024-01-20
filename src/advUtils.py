import muspy
import pathlib
import src.representation as representation
import numpy as np
import src.dataset as dataset
import matplotlib.pyplot as plt
import torch
import tqdm

#DATA Loading Utility Functions

def adjust_resolution(music, resolution):
    """Adjust the resolution of the music."""
    music.adjust_resolution(resolution)
    for track in music:
        for note in track:
            if note.duration == 0:
                note.duration = 1
    music.remove_duplicate()

def extract_json(midi_dir,json_dir=False,skip_existing=True,resolution=12):
    """Converts midi files into muspy json files"""
    if(not json_dir):
        json_dir=midi_dir
    json_dir = pathlib.Path(json_dir)
    for file in pathlib.Path(midi_dir).glob("*.mid"):
        out_name = file.name
        out_filename = json_dir / f"{out_name}.json"
        # Skip if the output file exists
        if skip_existing and out_filename.is_file():
            continue
        # Read the MIDI file
        music = muspy.read(file)
        # Adjust the resolution
        adjust_resolution(music, resolution)
        # Filter bad files
        end_time = music.get_end_time()
        if end_time > resolution * 4 * 2000 or end_time < resolution * 4 * 10:
            continue
        # Save as a MusPy JSON file
        out_filename.parent.mkdir(exist_ok=True, parents=True)
        music.save(out_filename)


def extract_repr(json_dir,repr_dir,skip_existing=True, resolution=12):
    """Converts json files into muspy npy and csv files"""
    for file in pathlib.Path(json_dir).glob("*.json"):
        out_name=file.name.split(".")[0]
        out_filename = repr_dir+"/"+f"{out_name}"
        outfilenpy = repr_dir+"/"+f"{out_name}.npy"
        outfilecsv = repr_dir+"/"+f"{out_name}.csv"
        if skip_existing and pathlib.Path(outfilenpy).is_file() or pathlib.Path(outfilecsv).is_file():
            continue
        # Load the score
        music = muspy.load(file)
        print(resolution,music.resolution)
        # Encode the score
        notes = representation.extract_notes(music, resolution)
        # Filter out bad files
        if len(notes) < 50:
            continue
        # Set start beat to zero
        notes[:, 0] = notes[:, 0] - notes[0, 0]
        # Make sure output directory exists
        pathlib.Path(out_filename).parent.mkdir(exist_ok=True)
        # Save the notes as a CSV file
        representation.save_csv_notes(outfilecsv, notes)
        # Save the notes as a NPY file
        np.save(outfilenpy, notes)

def convert_extract_load(train_args,encoding,midi_dir=False,json_dir=False,repr_dir=False, resolution=12, skip_existing=True, use_csv=False):
    """Creates a Dataset from the files in repr_dir, given a folder of Midi or json files, it will fist convert and add them to the given repr_folder"""
    # Get output 
    if(midi_dir):
        extract_json(midi_dir,json_dir=json_dir,skip_existing=skip_existing,resolution=resolution)
    if(json_dir):
        extract_repr(json_dir,repr_dir,skip_existing=skip_existing,resolution=resolution)
    if(repr_dir):
        if(use_csv):
            names = [i.name.split(".")[0] for i in list(pathlib.Path(repr_dir).glob("*.csv"))]
        else:
            names = [i.name.split(".")[0]+"\n" for i in list(pathlib.Path(repr_dir).glob("*.npy"))]
        with open(repr_dir+"/names.txt","w") as file:
            file.writelines(names)
            file.close()
        test_dataset = dataset.MusicDataset(
            repr_dir+"/names.txt",
            repr_dir,
            encoding,
            max_seq_len=train_args["max_seq_len"],
            max_beat=train_args["max_beat"],
            use_csv=use_csv,
        )
        return test_dataset

#UTILS for data writing
def save_pianoroll(filename, music, size=None, **kwargs):
    """Save the piano roll to file."""
    music.show_pianoroll(track_label="program", **kwargs)
    if size is not None:
        plt.gcf().set_size_inches(size)
        plt.savefig(filename)
        plt.close()


def save_result(filename, data, sample_dir, encoding, savenpy=True,savecsv=True,savetxt=True,savejson=True,savemid=True,savepng=True):
    """Save the results in multiple formats!"""
    pathlib.Path(sample_dir).mkdir(exist_ok=True)
    if(savenpy):
        # Save as a numpy array
        pathlib.Path(sample_dir+"/npy").mkdir(exist_ok=True)
        np.save(sample_dir+"/npy/"+f"{filename}.npy", data)
    if(savecsv):
    # Save as a CSV file
        pathlib.Path(sample_dir+"/csv").mkdir(exist_ok=True)
        representation.save_csv_codes(sample_dir+"/csv/"+f"{filename}.csv", data)
    if(savetxt):
         # Save as a TXT file
        pathlib.Path(sample_dir+"/txt").mkdir(exist_ok=True)
        representation.save_txt(
            sample_dir+"/txt/"+f"{filename}.txt", data, encoding
        )
    if(savejson or savemid):
        # Convert to a MusPy Music object
        music = representation.decode(data, encoding)
    if(savejson):
        # Save as a MusPy JSON file
        pathlib.Path(sample_dir+"/json").mkdir(exist_ok=True)
        music.save(sample_dir+"/json/"+f"{filename}.json")
    if(savepng):
        # Save as a piano roll
        pathlib.Path(sample_dir+"/png").mkdir(exist_ok=True)
        save_pianoroll(
            sample_dir+"/png/"+f"{filename}.png", music, (20, 5), preset="frame"
        )
    if(savemid):
        # Save as a MIDI file
        pathlib.Path(sample_dir+"/mid").mkdir(exist_ok=True)
        music.write(sample_dir+"/mid/"+f"{filename}.mid")


# Generating Functions:
def generate(n,sample_dir,model,test_loader,encoding,device,modes=["unconditioned"],seq_len=1024,temperature=1,filter_logits="top_k",filter_thresh=0.9):
    with torch.no_grad():
        sos = encoding["type_code_map"]["start-of-song"]
        eos = encoding["type_code_map"]["end-of-song"]
        beat_0 = encoding["beat_code_map"][0]
        beat_4 = encoding["beat_code_map"][4]
        beat_16 = encoding["beat_code_map"][16]
        data_iter = iter(test_loader)
        for i in tqdm.tqdm(range(n), ncols=80):
            batch = next(data_iter)
            print("Generating based on",batch['name'])
            for mode in modes:
                if(mode=="unconditioned"):
                    tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
                    tgt_start[:, 0, 0] = sos
                elif(mode=="instrument_informed"):
                    prefix_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_0))
                    tgt_start = batch["seq"][:1, :prefix_len].to(device)
                elif(mode=="4_beat"):
                    cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_4))
                    tgt_start = batch["seq"][:1, :cond_len].to(device)
                elif(mode=="16_beat"):
                    cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_16))
                    tgt_start = batch["seq"][:1, :cond_len].to(device)
                # Generate new samples
                generated = model.generate(
                    tgt_start,
                    seq_len,
                    eos_token=eos,
                    temperature=temperature,
                    filter_logits_fn=filter_logits,
                    filter_thres=filter_thresh,
                    monotonicity_dim=("type", "beat"),
                )
                generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

                # Save the results
                save_result(
                    f"{i}_{mode}", generated_np[0], sample_dir, encoding,savecsv=False,savetxt=False,savenpy=False,savepng=False,savejson=False
                )