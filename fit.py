#!/home/schoen/Scripts/perseus/ThirdParty/Python/bin/python
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
#import dipy.data as dpd
reload(dti)

import matplotlib.pyplot as plt

Version = "0.1"

doc = """
Model Fitting, Version {0}.

Usage:
    orientationcheck.py [options] --image=<FILE> --bval=<FILE> --bvec=<FILE> --outprefix=<FILE> [(--axis s | --axis c | --axis a) --slice <slice>] --fit_type

Options:
    -h --help          Show this screen.
    -v --version       Show version.
    --image=<FILE>     Image File (path).
    --bval=<FILE>      BVAL File (path).
    --bvec=<FILE>      BVEC File (path).
    --out=<FILE>       Output File (path).
    --slice <slice>    Specify a slice [default: None]
    --fit_type <Type>  Specify a type (LS, NLLS, RT)[default: WLS]
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

    ## the fit method will take in the data, regardless of shape.
    print("Fitting {0} data.".format(view))
    fit_wls = dti_wls.fit(image)

    return (fit_wls)


def run(rawargs):
    #This gets us our arguments. May need to be tweaked for the new ones we are adding.
    arguments = docopt(doc, argv=rawargs, version='Orientation Check v{0}'.format(Version))
    inputs = [{"Value":"image file", "Flag": "--image"}, {"Value":"bvec file", "Flag": "--bvec"}, {"Value":"bvec file", "Flag": "--bvec"}]
    for inputinfo in inputs:
        if not exists(arguments[inputinfo["Flag"]]):
            print("The {0} specified does not exist!".format(inputinfo["Value"]))
            sys.exit(1)


    #Load image. Always do this.
    image = nib.load(arguments["--image"])

    #Create gradient table. Always do this.
    print("Generating gradient table.")
    gtab = gradient_table(read_bvals_bvecs(arguments['--bval'], arguments['--bvec']))

    #### MOVED ABOVE >>>>>
    #Define the tensor model
    # print("Generating the tensor model.")
    # dti_wls = dti.TensorModel(gtab, fit_method="NLLS")
    #### MOVED ABOVE <<<<<


    #Keep this here
    print("Masking the brain.")
    image_masked, mask = median_otsu(image_data, 3, 1, autocrop=True, dilate=2)
    #If the user provides a mask, we could try to mask with it instead

    #Not necessary
    # print("Resampling the brain to a standard resolution.")
    # image, affine1 = reslice(image_masked, image.get_affine(), image.get_header().get_zooms()[:3], (3.0,3.0,3.0))

    print("Checking the image dimensions")
    Xsize, Ysize, Zsize, directions = image.shape
    print("X: {0}\nY: {1}\nZ: {2}".format(Xsize, Ysize, Zsize))
    #If a slice is specified, it might be a good idea to double-check that the slice is a valid index.
    #If so, we will extract that slice from the image.

    #Define Image Scopes. This will need to be adjusted to grab the slice at the given index, not the middle one.
    print("Defining the image scopes.")
    imagedict = {"axial": {"dropdim": [0,1], "scope": (slice(0,Xsize), slice(0,Ysize), slice(math.floor(Zsize/2),math.floor(Zsize/2)+1))},
                 "coronal": {"dropdim": [0,2], "scope": (slice(0,Xsize), slice(math.floor(Ysize/2),math.floor(Ysize/2)+1), slice(0, Zsize))},
                 "sagittal": {"dropdim": [1,2], "scope": (slice(math.floor(Xsize/2),math.floor(Xsize/2)+1), slice(0,Ysize), slice(0, Zsize))}}


    #roi_idx = (slice(0,image.shape[0]), slice(0,image.shape[1]), slice(middleslice,middleslice+1))#(slice(0,image.shape[0]), slice(0,image.shape[1]), slice(int(image.shape[2]/2),int(image.shape[2]/2)+1))
    # print("Defining sphere.")
    # sphere = dpd.get_sphere('symmetric724')

    #Slice the whole dataset by the scope
    if arguments["--slice"] == None:
        result = fit(image, gtab, fit_type=arguments["--fit"])
    else:
        result = fit(image[imagedict[arguments["--axis"]["scope"]]], gtab, fit_type=arguments["--fit"])

    #Saving data will be done here. The rest of the stuff below is unnecessary.

    #print("Slicing the dataset with the scopes.")
    #data_to_fit = image[imagedict[view]["scope"]]
    # for view in ["sagittal", "coronal", "axial"]:
    #     imagedict[view]["image"] = image[imagedict[view]["scope"]]

        # print("Fitting {0} data.".format(view))
        # fit_wls = dti_wls.fit(imagedict[view]["image"])

        # print("Extracting {0} FA.".format(view))
        # fa1 = fit_wls.fa
        # print("Extracting {0} EVALS.".format(view))
        # evals1 = fit_wls.evals
        # print("Extracting {0} EVECS.".format(view))
        # evecs1 = fit_wls.evecs
        # print("Extracting {0} Color FA.".format(view))
        # cfa1 = dti.color_fa(fa1, evecs1)
        # cfa1 = cfa1/cfa1.max()
        # print("Defining {0} renderer.".format(view))
        # render = fvtk.ren()
        # print("Generating {0} image.".format(view))
        # x =cfa1.shape[imagedict[view]["dropdim"][0]]
        # y =cfa1.shape[imagedict[view]["dropdim"][1]]
        #
        # #print(x, y, 1, 3)
        # cfa2 = cfa1.reshape(x, y, 1, 3)
        # evals2 = evals1.reshape(x, y, 1, 3)
        # evecs2 = evecs1.reshape(x, y, 1, 3, 3)
        #
        # fvtk.add(render, fvtk.tensor(evals2, evecs2, cfa2, sphere))
        # fvtk.record(render, out_path=arguments["--outprefix"]+"_"+view+".png", size=(800,800), magnification=2)
        # print("Image Saved")
    sys.exit(0)

if __name__ == '__main__':
    args = sys.argv
    del args[0]
    run(args)
