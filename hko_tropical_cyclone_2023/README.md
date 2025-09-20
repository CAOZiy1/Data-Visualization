# Typhoon Best Track Visualization

This project visualizes typhoon best track data from a CSV file on a world map. Each typhoon's path is colored by wind speed or intensity, with a focus on visual appeal for artists and designers.

## Features
- Reads typhoon best track data from CSV
- Plots each typhoon's path on a world map
- Colors paths by wind speed or intensity
- Uses matplotlib and cartopy for visualization

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Place the CSV file (e.g., HKO2023BST.csv) in the project directory
3. Run the script: `python main.py`

## Requirements
- Python 3.8+
- matplotlib
- cartopy
- pandas
- numpy

## Sample Output
The script will generate a file named `typhoon_tracks.png` in the project directory, showing the typhoon paths colored by wind speed or intensity on a world map. Open this file to view the visualization.
