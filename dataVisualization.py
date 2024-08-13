from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np

data = pd.read_csv('hand_landmarks.csv')
#letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
letters = ['b', 'd', 'f', 'k']

def plotLetterData(data, letter):
    # get all rows with the parameter 'letter'
    letter_data = data[data['label'] == letter]
    
    plt.figure(figsize=(24, 16))

    color_map = plt.cm.get_cmap('tab20')

    for i, (_, row) in enumerate(letter_data.iterrows()):
        # Extract x and y coordinates
        x_coords = [row[f'{j}_x'] for j in range(21)]
        y_coords = [row[f'{j}_y'] for j in range(21)]
        
        # Plot the points and lines
        color = color_map(i / len(letter_data))
        plt.plot(x_coords, y_coords, '-o', color=color, label=f'Instance {i+1}')
    
    plt.title(f'Hand Landmark Coordinates for Letter {letter}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

for letter in letters:
    plotLetterData(data, letter)
