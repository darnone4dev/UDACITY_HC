import sys
import os
import pydicom
if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    path = '/data/TestVolumes/Study1/13_HCropVolume/19.dcm'
    if len(sys.argv) == 2:
        path = sys.argv[1]

    print(f'Load dicom file {path}') 

    dicom = pydicom.dcmread(path)
    print(dicom)