#!../Python/bin/python
import matplotlib.pyplot as plt
import os, sys, pandas, math
import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
from dipy.viz import fvtk
from dipy.segment.mask import median_otsu
from dipy.align.reslice import reslice
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from docopt import docopt
#import dipy.data as dpd
reload(dti)

Version = "0.1"

doc = """
Orientation Check, Version {0}.

Usage:
    fit.py [options] --image=<FILE> --bval=<FILE> --bvec=<FILE> --outprefix=<FILE>
    fit.py [options] --image=<FILE> --bval=<FILE> --bvec=<FILE> --outprefix=<FILE> --axis=<type> --slice=<slice>

Options:
    -h --help            Show this screen.
    -v --version         Show version.
    --image=<FILE>       Image File (path).
    --image_mask=<FILE>  Image Mask File (path). [default: None]
    --bval=<FILE>        BVAL File (path).
    --bvec=<FILE>        BVEC File (path).
    --outprefix=<FILE>   Output File (path).
    --slice=<slice>      Specify a slice [default: None]
    --axis=<type>        Specify an axis (coronal, sagittal, axial) [default: coronal]
    --fit_type=<type>    Specify a type (WLS, LS, NLLS, RT) [default: WLS]
""".format(Version)

#============================================================================
#             General Utility
#============================================================================

def cleanPathString(path):
  if path.endswith("/"):
    path = path[:-1]
  if path.startswith("="):
    path = path[1:]
  realpath = os.path.realpath(path)
  return realpath


def exists(path):
    #Shorthand for checking the existence of a file
    if os.path.exists(cleanPathString(path)):
        return(1)
    else:
        print("Error: Input file '{0}' not found!".format(path))
        return(0)

def fit(image, mask, gtab, fit_type="WLS"):
    print("Generating the tensor model.")
    dti_model = dti.TensorModel(gtab, fit_method=fit_type)

    #the fit method will take in the data, regardless of the shape
    print("Fitting data.")
    fitted = dti_model.fit(image, mask=mask)
    # print("Generating prediction.")
    # prediction = dti_model.predict()
    return fitted

def pad_val(val, pad_length=3):
    if type(val) == str:
        return val.zfill(pad_length)
    elif type(val) == int:
        replace_string = "{0:0"+str(pad_length)+"d}"
        return replace_string.format(val)

def run(rawargs):
    #retrieves arguments
    arguments = docopt(doc, argv=rawargs, version='Orientation Check v{0}'.format(Version))
    #print(arguments)
    inputs = [{"Value":"image file", "Flag": "--image"}, {"Value":"bvec file", "Flag": "--bvec"}, {"Value":"bvec file", "Flag": "--bvec"}]
    if arguments["--image_mask"] != None and arguments["--image_mask"] != 'None':
        inputs.append({"Value":"image mask file", "Flag": "--image_mask"})
    for inputinfo in inputs:
        if not exists(arguments[inputinfo["Flag"]]):
            print("The {0} specified does not exist!".format(inputinfo["Value"]))
            sys.exit(1)

    #Load image
    image = nib.load(arguments["--image"])
    image_data = image.get_data()

    #Generate gradient table
    print("Generating gradient table.")
    bvals, bvecs = read_bvals_bvecs(arguments['--bval'], arguments['--bvec'])
    gtab = gradient_table(bvals, bvecs)


    #Generate an image that is an average of all b0 volume (useful for reverse prediction)
    print("Generating signal intensity map.")
    values = np.array(bvals)
    ii = np.where(values == bvals.min())[0]
    image_b0_average = np.mean(image_data[:,:,:,ii], axis=3)

    print("Masking the brain.")
    if arguments["--image_mask"] != None and arguments["--image_mask"] != 'None':
        #Mask the image with the mask provided
        image_mask = nib.load(arguments["--image_mask"])
        image_mask_data = image_mask.get_data()
        #THE FOLLOWING DOES NOT WORK: image_data is 4D, and image_mask is likely 3D. Will need to mask differently
        image_masked = image_mask_data * image_data
    else:
        image_masked, image_mask = median_otsu(image_data, 3, 1, autocrop=False, dilate=2)
        image_b0_average_masked = image_mask * image_b0_average

    print("Checking the image dimensions")
    Xsize, Ysize, Zsize, directions = image.shape
    print("X: {0}\nY: {1}\nZ: {2}".format(Xsize, Ysize, Zsize))
    #If a slice is specified, double-check that the slice is a valid index
    #If it is valid, extract that slice from the image


    if arguments["--slice"] == None or arguments["--slice"] == 'None':
        #Fit the data
        result = fit(image_data, image_mask, gtab, fit_type=arguments["--fit_type"])

        #Generate average signal, to be used in reverse calculation
        image_average_signal = np.mean(image_masked, axis=3)

        #Define the paths to the output files
        spd_file_path = arguments["--outprefix"]+'_spd.nii.gz'
        estimate_file_path = arguments["--outprefix"]+'_ecc_estimated.nii.gz'
        error_file_path = arguments["--outprefix"]+'_ecc_error.nii.gz'

        #Generate the SPD data (lower triangular of the symmetric matrix)
        spd_data = result.lower_triangular()
        #Predict the original data based on the resulting tensor data
        estimate_data = result.predict(gtab, S0=image_b0_average_masked)
        #Generate the difference between original and the predicted original
        error_data = np.absolute(image_masked - estimate_data)


    else:
        #Force the argument to be in integer form
        arguments["--slice"] = int(arguments["--slice"])

        #Do some checking on the boundaries. Exit if not in bounds.
        if arguments["--axis"] == "sagittal":
            axisbound = Xsize
        elif arguments["--axis"] == "coronal":
            axisbound = Ysize
        else:
            axisbound = Zsize
        if 0 > arguments["--slice"] or arguments["--slice"] > axisbound:
            raise IOError("Slice does not exist on given axis.")
            sys.exit(1)

        #Slice all the datasets required
        if arguments["--axis"] == "axial":
            image_data_slice = image_data[:,:,arguments["--slice"]]
            image_mask_slice = image_mask[:,:,arguments["--slice"]]
        elif arguments["--axis"] == "sagittal":
            image_data_slice = image_data[:,arguments["--slice"],:]
            image_mask_slice = image_mask[:,arguments["--slice"],:]
        else:
            image_data_slice = image_data[arguments["--slice"],:,:]
            image_mask_slice = image_mask[arguments["--slice"],:,:]

        #Fit the data
        result = fit(image_data_slice, image_mask_slice, gtab, fit_type=arguments["--fit_type"])

        #Generate average signal, to be used in reverse calculation
        image_average_signal_slice = np.mean(image_data_slice*image_mask_slice, axis=3)

    #print(result.lower_triangular())
    # if arguments["--slice"] == None or arguments["--slice"] == 'None':
    #
    # else:
    #     spd_file_path = arguments["--outprefix"]+arguments["--axis"]+pad_val(arguments["--slice"])+'_spd.nii.gz'
    #     spd_data = result.lower_triangular()
    #     estimate_file_path = arguments["--outprefix"]+arguments["--axis"]+pad_val(arguments["--slice"])+'_ecc_estimated.nii.gz'
    #     #THE FOLLOWING DOES NOT WORK: image_mask is 3D, and result.predict(gtab) produces a 4D dataset. Will need to mask differently
    #     estimate_data = result.predict(gtab, S0=image_average_signal_slice)# * image_mask
    #     error_file_path = arguments["--outprefix"]+arguments["--axis"]+pad_val(arguments["--slice"])+'_ecc_error.nii.gz'
    #     error_data = numpy.absolute(image_masked[imagedict[arguments["--axis"]]["scope"]] - estimate_data)

    print("Saving SPD image to "+spd_file_path)
    nib.save(nib.Nifti1Image(spd_data, image.get_affine()), spd_file_path)
    print("Saving estimated image to "+estimate_file_path)
    nib.save(nib.Nifti1Image(estimate_data, image.get_affine()), estimate_file_path)
    print("Saving error image to "+error_file_path)
    nib.save(nib.Nifti1Image(error_data, image.get_affine()), error_file_path)

    #nib.save(nib.Nifti1Image(result.lower_triangular(), image.get_affine()), arguments["--outprefix"]+'_spd.nii.gz')



    sys.exit(0)

if __name__ == '__main__':
    args = sys.argv
    del args[0]
    run(args)
