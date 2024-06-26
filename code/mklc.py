import numpy as np
import matplotlib.pyplot as plt
import os
import random

#for colored graphs
from matplotlib.colors import Normalize
from matplotlib.pyplot import get_cmap

from tqdm import tqdm

def linelc(arr, x, y, lcdir):
    flxxy = arr[x, y]

    # Generate the z indices
    z_indices = np.arange(flxxy.shape[0])

    #Plot the data
    plt.figure(figsize=(8, 2), dpi=300)
    plt.plot(z_indices, flxxy, linestyle='-', marker='')
    plt.xlabel('Cadence Number')
    plt.ylabel('Flux')
    #plt.ylim(-10, 10)
    #plt.xlim(2000, 4000)
    plt.title(f'Cadence vs. Malena Subtraction at ({x}, {y})')
    plt.grid(True)

    finnme = f"madiff_{x}_{y}.png"

    output = os.path.join(lcdir, finnme)
    plt.savefig(output)
    plt.close()
    #print(f"Plot saved as {output}")

def scatterlc(arr, lcdir):
    pltnum = 200
    random_coordinates = [(random.randint(125, 149), random.randint(0, 149)) for _ in range(pltnum)]

    # Generate the z indices
    z_indices = np.arange(arr.shape[2])

    #normalize coords to fit colormap
    norm = Normalize(vmin=0, vmax=149)
    cmap = plt.get_cmap('cool')

    #Plot the data
    plt.figure(figsize=(7.5, 2.25), dpi = 300)
    for ii in tqdm(range(pltnum)):
        coord = random_coordinates[ii]
        color = cmap(norm(coord[1]))
        plt.plot(z_indices, arr[coord[0], coord[1], :], color=color, linestyle = '-', marker=' ')
    
    plt.xlabel('Cadence Number')
    plt.ylabel('Flux')
    plt.ylim(-250, 10000)
    #plt.xlim(2000, 4000)
    plt.title(f'Cadence vs. Malena Subtraction over {pltnum} pixels')
    plt.grid(True)

    finnme = f"mahandful.png"

    output = os.path.join(lcdir, finnme)
    plt.savefig(output)
    plt.close()

pltnum = 5
random_coordinates = [(114, 113), (51, 65), (79, 134), (89, 30)]#[(random.randint(0, 149), random.randint(0, 149)) for _ in range(pltnum)]
array = np.load('/scratch11/ktp9/DIA/70/stacks/malenasubtract.npy')
lcdir = '/scratch11/ktp9/DIA/70/lcfiles/'

scatterlc(array, lcdir)
#for x, y in tqdm(random_coordinates):
    #linelc(array, x, y, lcdir)
print("Here's your plot! Bye-bye buttrefly!")
