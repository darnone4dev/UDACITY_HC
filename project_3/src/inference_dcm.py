"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess


import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """
    if not dcmlist or len(dcmlist)==0:
        raise ValueError('error: dcmlist empty')
    # In the real world you would do a lot of validation here
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ignored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """

    # TASK: Compute the volume of your hippocampal prediction
    # <YOUR CODE HERE>
    # We compute the number of voxels , but can't know the real volume       # within the 
    volume_ant = get_voxels(pred,1)
    volume_post = get_voxels(pred,2)
    total_volume = get_voxels(pred,0) + volume_ant + volume_post
    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def get_voxels(image, pickup):
    return(np.sum(image == pickup))

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    print('prepare report...')
    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)
    header_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=20)          
    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4] # is there a better choice?

    # TASK: Create the report here and show information that you think would be relevant to
    # clinicians. A sample code is provided below, but feel free to use your creative 
    # genius to make if shine. After all, the is the only part of all our machine learning 
    # efforts that will be visible to the world. The usefulness of your computations will largely
    # depend on how you present them.

    # SAMPLE CODE BELOW: UNCOMMENT AND CUSTOMIZE
    #      import pydicom

    # Access basic patient information
    #patient_name = header.PatientName
    print('retrieving dicom metadata....')
    patient_name = header[(0x0010,0x0010)].value
    patient_id = header[(0x0010,0x0020)].value
    #patient_birth_date = header.PatientBirthDate
    #patient_sex = header.PatientSex
    patient_sex =''
    #patient_age = header.PatientAge
    patient_age = ''

    # Access study information
    dt = header[(0x0008,0x0020)]
    study_date = dt[:4] + '-' + dt[4:6] + '-' + dt[6:8]
    print(study_date)
    dt = header[(0x0008,0x0031)]
    study_time =  dt[:2] + ':' + dt[2:4] + ':' + dt[4:]
    print(study_time)
    #accession_number = header.AccessionNumber
    accession_number = header[(0x0008,0x0050)].value
    study_id = header[(0x0020,0x0010)].value
    #referring_physician = header.ReferringPhysicianName

    # Access series information
    #series_description = header.SeriesDescription
    series_description = ''

    #modality = header.Modality
    modality = header[(0x0008,0x0060)].value
    #body_part_examined = header.BodyPartExamined

    # Access image information
    #image_position = header.ImagePositionPatient
    image_position = header[(0x0020,0x0032)].value
    #image_orientation = header.ImageOrientationPatient
    image_orientation = header[(0x0020,0x0037)].value

    #pixel_spacing = header.PixelSpacing
    pixel_spacing = header[(0x0028,0x0030)].value
    #slice_thickness = header.SliceThickness
    slice_thickness = header[(0x0018,0x0050)].value

    
    # Access other relevant information
    #manufacturer = header.Manufacturer
    #institution_name = header.InstitutionName
    #software_versions = header.SoftwareVersions

    draw.text((10, 10), "\n\nHippoVolume.AI", (255, 255, 255), font=header_font)
    
    report_content = f"\n\n\n\n\nSTUDY INFO\n"\
                     f"Acces number: {accession_number},Study date: {study_date} ,Study time: {study_time}\n"\
                     f"SERIE INFO\n"\
                     f"Modality: {modality}\n"\
                     f"IMAGE INFO\n"\
                     f"Image position: {image_position} Image orientation: {image_orientation} Pixel spacing: {pixel_spacing} Slice thickness: {slice_thickness}" 
                       
    draw.multiline_text((10, 30),report_content,
                         (255, 255, 255), font=main_font)

    # STAND-OUT SUGGESTION:
    # In addition to text data in the snippet above, can you show some images?
    # Think, what would be relevant to show? Can you show an overlay of mask on top of original data?
    # Hint: here's one way to convert a numpy array into a PIL image and draw it inside our pimg object:
    #
    # Create a PIL image from array:
    # Numpy array needs to flipped, transposed and normalized to a matrix of values in the range of [0..255]

    for i in range(len(slice_nums)):
        print(f'include images and prediction {i}')
        orig_slice = orig_vol[:,:,slice_nums[i]]
        pred_slice = pred_vol[:,:,slice_nums[i]]  
        nd_orig_img = np.flip((orig_slice/np.max(orig_slice))*0xff).T.astype(np.uint8)
        nd_pred_img = np.flip((pred_slice/np.max(pred_slice))*0xff).T.astype(np.uint8)
          
        # This is how you create a PIL image from numpy array
        pil_orig_img = Image.fromarray(nd_orig_img, mode="L").convert("RGBA").resize((128,128))
        pil_pred_img = Image.fromarray(nd_pred_img, mode="L").convert("RGBA").resize((128,128))

        # Paste the PIL image into our main report image object (pimg)
        pimg.paste(pil_orig_img, box=(10, 300 + i * 128))
        pimg.paste(pil_pred_img, box=(200, 300 + i * 128))
          
    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # Code below creates a DICOM Secondary Capture instance that will be correctly
    # interpreted by most imaging viewers including our OHIF
    # The code here is complete as it is unlikely that as a data scientist you will 
    # have to dive that deep into generating DICOMs. However, if you still want to understand
    # the subject, there are some suggestions below

    # Set up DICOM metadata fields. Most of them will be the same as original file header
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # STAND OUT SUGGESTION: 
    # If you want to understand better the generation of valid DICOM, remove everything below
    # and try writing your own DICOM generation code from scratch.
    # Refer to this part of the standard to see what are the requirements for the valid
    # Secondary Capture IOD: http://dicom.nema.org/medical/dicom/2019e/output/html/part03.html#sect_A.8
    # The Modules table (A.8-1) contains a list of modules with a notice which ones are mandatory (M)
    # and which ones are conditional (C) and which ones are user-optional (U)
    # Note that we are building an RGB image which would have three 8-bit samples per pixel
    # Also note that writing code that generates valid DICOM has a very calming effect
    # on mind and body :)

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    print(path)
    dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

    # TASK: create a series_for_inference variable that will contain a list of only 
    # those PyDicom objects that represent files that belong to the series that you 
    # will run inference on.
    # It is important to note that radiological modalities most often operate in terms
    # of studies, and it will most likely be on you to establish criteria for figuring 
    # out which one of the multiple series sent by the scanner is the one you need to feed to 
    # your algorithm. In our case it's rather easy - we have reached an agreement with 
    # people who configured the HippoCrop tool and they label the output of their tool in a 
    # certain way. Can you figure out which is that? 
    # Hint: inspect the metadata of HippoCrop series

    # <YOUR CODE HERE>
    series_for_inference = []
    for dicom in dicoms:
        seriesDescription = dicom.SeriesDescription
        modality = dicom.Modality
        #filter for our Hipocampal algorthim
        if modality == 'MR' and seriesDescription == 'HippoCrop':
           series_for_inference.append(dicom) 
    
    #series_for_interference = 

    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def os_command(command):
    # Comment this if running under Windows
    print('os_command')
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

    # Uncomment this if running under Windows
    # os.system(command)

if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]

    # Get the latest directory
    # study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]


    # TASK: get_series_for_inference is not complete. Go and complete it
    for study_dir in subdirs:
        try:
            print(f"Looking for series to run inference on in directory {study_dir}...")
            volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
            print(f"Found series of {volume.shape[2]} axial slices")

            print("HippoVolume.AI: Running inference...")
            # TASK: Use the UNetInferenceAgent class and model parameter file from the previous section
            inference_agent = UNetInferenceAgent(
                device="cpu",
                parameter_file_path=r"/home/workspace/out/model/model.pth")
            # Run inference
            # TASK: single_volume_inference_unpadded takes a volume of arbitrary size 
            # and reshapes y and z dimensions to the patch size used by the model before 
            # running inference. Your job is to implement it.
            pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
            # TASK: get_predicted_volumes is not complete. Go and complete it
            pred_volumes = get_predicted_volumes(pred_label)
            # Create and save the report
            print("Creating and pushing report...")
            report_save_path = f"/home/workspace/out/studies/report_{header.SeriesDescription}.dcm"
            # TASK: create_report is not complete. Go and complete it. 
            # STAND OUT SUGGESTION: save_report_as_dcm has some suggestions if you want to expand your
            # knowledge of DICOM format
            print(f'save report {report_save_path} ....')
            report_img = create_report(pred_volumes, header, volume, pred_label)
            save_report_as_dcm(header, report_img, report_save_path)
            print(f'report {report_save_path} saved')

            # Send report to our storage archive
            # TASK: Write a command line string that will issue a DICOM C-STORE request to send our report
            # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    
            orthanc_cmd = f"storescu 127.0.0.1 4242 -v -aec HIPPOAI {report_save_path}"
            print(f"execute command {orthanc_cmd}...")
            os_command(orthanc_cmd)
            print(f"command {orthanc_cmd} executed")
            # This line will remove the study dir if run as root user
            # Sleep to let our StoreSCP server process the report (remember - in our setup
            # the main archive is routing everyting that is sent to it, including our freshly generated
            # report) - we want to give it time to save before cleaning it up
            time.sleep(2)
            shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

            print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
                f"volume ant: {pred_volumes['anterior']}, ",
                f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
        except Exception as e:
            # Handle any other exception
            print(f"An unexpected error occurred: {e}")