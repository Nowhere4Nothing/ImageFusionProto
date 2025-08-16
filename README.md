# ImageFusionProto
Making the protype for the image fusion for assessment 4

# Instructions for running
#   Windows
1. With a powershell window open on this folder with admin access, run
```
poetry --version # to ensure poetry is installed
python -m venv .venv 
.venv/Scripts/activate
```
This will then access a virtual environment to gather the dependencies required. It should now say **(.venv) PS C:/...**

2. Now to install the dependencies run:
```
poetry install
```
All the dependencies will now install in the venv.

3. Once all dependencies are installed, run:
```
python main.py
```

#   Unix
1. With a terminal window open on this folder with sudo access, run
```
poetry --version # to ensure poetry is installed
python3 -m venv .venv 
source .venv/bin/activate
```
This will then access a virtual environment to gather the dependencies required. It should now say **(imagefusionproto-py3.1x)...**

2. Now to install the dependencies run:
```
poetry install
```
All the dependencies will now install in the venv.

3. Once all dependencies are installed, run:
```
python3 main.py
```

# Usage Instructions
To dislay the data
1. Click the load DICOM Folder button
2. Navigate to the data folder
3. Press the Select the folder button on the data just before the data. Note: If you click in till it says "No items match your search" go back up one directory and select that folder.
4. Data should be loaded
5. To select a second layer follow the above steps again
6. Once you have two layers, select the overlay and make as transparent as needed via the opacity slider
7. Use the translate / rotate buttons to align images

# OUO
## To Do:
1. Optimise the rotation [MATT]
2. Add colour fusion (purple/green) (simple toggle button) [WILL]
3. Able to extract transformation objects from DICOM files and apply them automatically [MATT]
4. Toggle dosage overlay [WILL]
5. Add reset button that resets all sliders [WILL]
6. Add translation sliders (XYZ) [WILL]
7. Look at auto fusion code and see how to integrate or what functions can be reused [BOTH]
8. Split up file into smaller files [WILL]

## Feature list
- XYZ sliders (Translation)
- LR, PA, IS sliders (Rotation)
- Transfer ROI between DICOM image sets
- Reset the sliders
- Toggle the overlay DICOM image
- Colour fusion (potential for multiple colour options such as [(purple = base layer, green = overlay) (purple-green, yellow-darkblue, red-lightblue)])

## Questions
1. How does the automatic fusion work in OnkoDICOM?
2. Where to put opacity slider and toggle layer button (switch?)

## possible rotation fix library:
https://vtk.org/doc/nightly/html/index.html
