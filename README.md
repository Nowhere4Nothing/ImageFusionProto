# ImageFusionProto
Making the protype for the image fusion for assessment 4

# To Do:
1. Optimise the rotation [MATT]
2. Add colour fusion (purple/green) (simple toggle button) [WILL]
3. Able to extract transformation objects from DICOM files and apply them automatically [MATT]
4. Toggle dosage overlay [WILL]
5. Add reset button that resets all sliders [WILL]
6. Add translation sliders (XYZ) [WILL]
7. Look at auto fusion code and see how to integrate or what functions can be reused [BOTH]

# Feature list
- XYZ sliders (Translation)
- LR, PA, IS sliders (Rotation)
- Transfer ROI between DICOM image sets
- Reset the sliders
- Toggle the overlay DICOM image
- Colour fusion (potential for multiple colour options such as [(purple = base layer, green = overlay) (purple-green, yellow-darkblue, red-lightblue)])

# Questions
1. How does the automatic fusion work in OnkoDICOM?
2. Where to put opacity slider and toggle layer button (switch?)
3. 

# possible rotation fix library:
https://vtk.org/doc/nightly/html/index.html