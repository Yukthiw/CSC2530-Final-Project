import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
'''
Thread pool won't parallelize CPU bound parts due to Global interpreter lock (GIL)
Process pool gives each worker it's own memory space and CPU.
'''
from tqdm import tqdm

if __name__ == '__main__':

    print("running")

    sys.path.append(os.path.dirname(__file__))
    from augment_pc import AugmentPointCloud
    apc = AugmentPointCloud()

    def safe_augment(input_file, output_file):
        try: 
            apc.augment(input_file, output_file)
        except Exception as e:
            print(f"Failed to processes {input_file}: {e}")

    def process_directory(input_dir, output_dir, max_workers=8):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # check this

        files = list(input_dir.iterdir()) # Get all files

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(files), desc="processing files") as progress_bar:
                for input_file in files:
                    # print(f"Current progress: {progress_bar.n}/{progress_bar.total}")
                    if input_file.is_file(): # ensure it's a file

                        # compute the output file name: 
                        # strip the extensions
                        file_path_without_all_suffixes = input_file.name.split('.')[0]
                        output_file = output_dir / file_path_without_all_suffixes

                        # check if the file already exists. if so, continue to the next one
                        if (output_file.with_suffix('.npy')).exists():
                            print(f"file {output_file} already exists, skipping...")
                            progress_bar.update(1)
                            print(f"Current progress: {progress_bar.n}/{progress_bar.total}")
                            continue
                        
                        def callback(future):
                            progress_bar.update(1)
                            print(f"Current progress: {progress_bar.n}/{progress_bar.total}")

                        future = executor.submit(safe_augment, input_file, output_file)
                        future.add_done_callback(callback)
        
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]   
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4 

    # input_directory = "/w/331/yukthiw/nuscenes/samples/LIDAR_TOP"
    # output_directory = "/w/246/willdormer/projects/CompImaging/augmented_pointclouds/samples/LIDAR_TOP"

    process_directory(input_directory, output_directory, max_workers=max_workers)

