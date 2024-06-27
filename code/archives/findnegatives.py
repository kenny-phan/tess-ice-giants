import numpy as np
from tqdm import tqdm 

def checkValues(file_path, chunk_size=50):
    # Load the data from the .npy file
    data = np.load(file_path)
    x, y, n = data.shape
    negative_coords = []
    # Check for negative values
    has_negatives = np.any(data < 0)

    if has_negatives: 
        for start_frame in tqdm(range(0, n, chunk_size)):
            end_frame = min(start_frame + chunk_size, n)
            chunk_data = data[:, :, start_frame:end_frame]

            negative_ind = np.argwhere(chunk_data < 0)
            
            for idx in negative_ind:
                negative_coords.append((idx[0], idx[1], start_frame + idx[2]))
        return negative_coords
    else: 
        print("No negativity here!")

array = '/scratch11/ktp9/DIA/70/stacks/malenasubtract.npy'
svefil = '/scratch11/ktp9/DIA/70/stacks/ms_negatives.txt'
values = checkValues(array)

with open(svefil, 'w') as output:
    for item in tqdm(values):
        output.write(f"{item}/n")

