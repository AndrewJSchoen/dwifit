#!../Python/bin/python
import matplotlib.pyplot as plt
import os, sys, pandas, math, numpy
import nibabel as nib
import dipy.reconst.dti as dti
from dipy.viz import fvtk
from dipy.segment.mask import median_otsu
from dipy.align.reslice import reslice
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from docopt import docopt
import dipy.data as dpd
reload(dti)

import matplotlib.pyplot as plt

Version = "0.1"

doc = """
Orientation Check, Version {0}.

Usage:
    orientationcheck.py [options] --image=<FILE> --bval=<FILE> --bvec=<FILE> --outprefix=<FILE> [(--axis=sagittal | --axis=coronal | --axis=axial) --slice=<slice>] --fit_type=<type>

Options:
    -h --help          Show this screen.
    -v --version       Show version.
    --image=<FILE>     Image File (path).
    --bval=<FILE>      BVAL File (path).
    --bvec=<FILE>      BVEC File (path).
    --out=<FILE>       Output File (path).
    --slice=<slice>    Specify a slice [default: None]
    --axis=<type>      Specify an axis [default: coronal]
    --fit_type=<type>  Specify a type (LS, NLLS, RT)[default: WLS]
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

def fit(image, gtab, fit_type="WLS"):
    print("Generating the tensor model.")
    dti_wls = dti.TensorModel(gtab, fit_method=fit_type)

    #the fit method will take in the data, regardless of the shape
    print("Fitting {0} data.".format(view))
    fit_wls = dti_wls.fit(image)

def run(rawargs):
    #retrieves arguments
    arguments = docopt(doc, argv=rawargs, version='Orientation Check v{0}'.format(Version))
    inputs = [{"Value":"image file", "Flag": "--image"}, {"Value":"bvec file", "Flag": "--bvec"}, {"Value":"bvec file", "Flag": "--bvec"}]
    for inputinfo in inputs:
        if not exists(arguments[inputinfo["Flag"]]):
            print("The {0} specified does not exist!".format(inputinfo["Value"]))
            sys.exit(1)

        #Load image
        image = nib.load(arguments["--image"])

        #Generates gradient table
        print("Generating gradient table.")
        gtab = gradient_table(read_bvals_bvecs(arguments['--bval'], arguments['--bvec']))

        print("Masking the brain.")
        image_masked, mask = median_otsu(image_data, 3, 1, autocrop=True, dilate=2)

        print("Checking the image dimensions")
        Xsize, Ysize, Zsize, directions = image.shape
        print("X: {0}\nY: {1}\nZ: {2}".format(Xsize, Ysize, Zsize))
        #If a slice is specified, double-check that the slice is a valid index
        #If it is valid, extract that slice from the image

        if arguments["--slice"] == None or arguments["--slice"] == 'None':
            result = fit(image, gtab, fit_type=arguments["--fit_type"])
        else:
            arguments["--slice"] = int(arguments["--slice"])
            if arguments["--axis"] == "sagittal":
                axisbound = Xsize
            elif arguments["--axis"] == "coronal":
                axisbound = Ysize
            else:
                axisbound = Zsize
            if 0 <= arguments["--slice"] <= axisbound:
                print("Defining the image scopes.")
                imagedict = {"axial": {"dropdim": [0,1], "scope": (slice(0,Xsize), slice(0,Ysize), slice(arguments["--slice"],arguments["--slice"]+1))},
                             "coronal": {"dropdim": [0,2], "scope": (slice(0,Xsize), slice(arguments["--slice"],arguments["--slice"]+1), slice(0, Zsize))},
                             "sagittal": {"dropdim": [1,2], "scope": (slice(arguments["--slice"],arguments["--slice"]+1), slice(0,Ysize), slice(0, Zsize))}}
                result = fit(image[imagedict[arguments["--axis"]["scope"]]], gtab, fit_type=argument["--fit_type"])
            else:
                print("Slice does not exist on given axis.")

    sys.exit(0)

if __name__ == '__main__':
    args = sys.argv
    del args[0]
    run(args)
