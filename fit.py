#!../Miniconda/bin/python
import os, sys, pandas, math
import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
import dipy.reconst.dki as dki
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
    fit.py [options] --image <FILE> --bvals <FILE> --bvecs <FILE>

Options:
    -h --help                   Show this screen.
    -v --version                Show version.
    --image <FILE>              Input image file (path).
    --mask <FILE>               Mask image in same coordinate space as image, optional. (path) [default: None]
    --slice <INT>               Specify a specific slice to fit (int) [default: None].
    --bvals <FILE>              BVALS file (path).
    --bvecs <FILE>              BVECS file (path).
    --out_dti <FILE>            Output the lower triangular of the dti data (path).[default: None]
    --out_dti_residual <FILE>   Output the residual of the dti model fitting (path).[default: None]
    --out_dki <FILE>            Output the lower triangular of the dki data (path).[default: None]
    --out_dki_residual <FILE>   Output the residual of the dki model fitting (path).[default: None]
    --out_dti_fa <FILE>         Output the FA from the dti model (path).[default: None]
    --out_dti_md <FILE>         Output the MD from the dti model (path).[default: None]
    --out_dti_rd <FILE>         Output the RD from the dti model (path).[default: None]
    --out_dti_ad <FILE>         Output the AD from the dti model (path).[default: None]
    --out_dki_fa <FILE>         Output the FA from the dki model (path).[default: None]
    --out_dki_md <FILE>         Output the MD from the dki model (path).[default: None]
    --out_dki_rd <FILE>         Output the RD from the dki model (path).[default: None]
    --out_dki_ad <FILE>         Output the AD from the dki model (path).[default: None]
    --fit_method <METHOD>       Specify a method for fitting (WLS, LS, NLLS, RT) [default: WLS]

""".format(Version)

#============================================================================
#             Classes
#============================================================================

class Fit(object):

    def __init__(self, data, mask, gradient_table, fit_method, out_dti=None, out_dti_residual=None, out_dki=None, out_dki_residual=None, out_dti_fa=None, out_dti_md=None, out_dti_rd=None, out_dti_ad=None, out_dki_fa=None, out_dki_md=None, out_dki_rd=None, out_dki_ad=None):
        self.raw_data = data
        self.mask = mask
        self.data = self.raw_data
        self.gradient_table = gradient_table

        #Fitted
        self.dti_fitted = None
        self.dki_fitted = None


        #Raw-Type values
        self.out_dti = None
        self.out_dti_path = out_dti
        self.out_dti_residual = None
        self.out_dti_residual_path = out_dti_residual
        self.out_dki = None
        self.out_dki_path = out_dki
        self.out_dki_residual = None
        self.out_dki_residual_path = out_dki_residual
        self.fit_method = fit_method

        #Outcome Measures
        self.out_dti_fa = None
        self.out_dti_fa_path = out_dti_fa
        self.out_dti_md = None
        self.out_dti_md_path = out_dti_md
        self.out_dti_rd = None
        self.out_dti_rd_path = out_dti_rd
        self.out_dti_ad = None
        self.out_dti_ad_path = out_dti_ad
        self.out_dki_fa = None
        self.out_dki_fa_path = out_dki_fa
        self.out_dki_md = None
        self.out_dki_md_path = out_dki_md
        self.out_dki_rd = None
        self.out_dki_rd_path = out_dki_rd
        self.out_dki_ad = None
        self.out_dki_ad_path = out_dki_ad

    def save(self):
        out_matrix = [
        [self.out_dti_path, self.out_dti],
        [self.out_dki_path, self.out_dki],
        [self.out_dti_residual_path, self.out_dti_residual],
        [self.out_dki_residual_path, self.out_dki_residual],
        [self.out_dti_fa, self.out_dti_fa],
        [self.out_dti_md, self.out_dti_md],
        [self.out_dti_rd, self.out_dti_rd],
        [self.out_dti_ad, self.out_dti_ad],
        [self.out_dki_fa, self.out_dki_fa],
        [self.out_dki_md, self.out_dki_md],
        [self.out_dki_rd, self.out_dki_rd],
        [self.out_dki_ad, self.out_dki_ad],
        ]
        for path, contents in out_matrix:
            if path != None and contents != None:
                print("Saving {0}".format(path))
                nib.nifti1.save(contents, path)

    def slice(self):
        pass

    def apply_mask(self):
        """
        If self.mask is not None, will mask the raw_data with the mask provided.
        If self.mask is None, median_otsu is used to generate those files.
        """
        if self.mask == None:
            print("Generating mask with median_otsu.")
            raw_data = self.raw_data.get_data()
            masked_data, mask = median_otsu(raw_data, 2,2)

            #Update the instance
            self.data = nib.nifti1.Nifti1Image(masked_data.astype(np.float32), self.raw_data.get_affine())
            self.mask = nib.nifti1.Nifti1Image(mask.astype(np.int_), self.data.get_affine())
        else:
            print("Masking data with provided mask.")
            raw_data = self.raw_data.get_data()
            mask_data = self.mask.get_data()
            masked_data = raw_data * mask_data

            #Update the instance
            self.data = nib.nifti1.Nifti1Image(masked_data.astype(np.float32), self.raw_data.get_affine())
        self.slice()


    def fit_dti(self):
        """
        Fits a dti model to the data
        """
        data = self.data.get_data()

        #Generate an average B0 image
        values = np.array(self.gradient_table.bvals)
        ii = np.where(values == self.gradient_table.bvals.min())[0]
        print("Generating average B0 image.")
        b0_average = np.mean(data[:,:,:,ii], axis=3)

        #Generate the tensor model
        print("Generating the dti model.")
        tenmodel = dti.TensorModel(self.gradient_table, fit_method=self.fit_method)
        print("Fitting the data.")
        self.dti_fitted = tenmodel.fit(data)

        #Generate the lower-triangular dataset
        print("Generating the lower-triangular data.")
        spd_data = self.dti_fitted.lower_triangular()
        self.out_dti = nib.nifti1.Nifti1Image(spd_data, self.data.get_affine())

        #Generate the residuals
        print("Estimating input data.")
        estimate_data = self.dti_fitted.predict(self.gradient_table, S0=b0_average)
        print("Calculating residuals.")
        residuals = np.absolute(data - estimate_data)
        self.out_dti_residual = nib.nifti1.Nifti1Image(residuals.astype(np.float32), self.data.get_affine())



    def fit_dki(self):
        """
        Fits a dki model to the data
        """
        data = self.data.get_data()

        #Generate an average B0 image
        values = np.array(self.gradient_table.bvals)
        ii = np.where(values == self.gradient_table.bvals.min())[0]
        b0_average = np.mean(data[:,:,:,ii], axis=3)

        #Generate the dk model
        dkimodel = dki.DiffusionKurtosisModel(self.gradient_table)
        self.dki_fitted = dkimodel.fit(data)

        #Generate the lower-triangular dataset
        spd_data = tenfit.lower_triangular()
        self.out_dti = nib.nifti1.Nifti1Image(spd_data, self.data.get_affine())

        #Generate the residuals
        estimate_data = dkimodel.predict(self.gradient_table, S0=b0_average)
        residuals = np.absolute(data - estimate_data)
        self.out_dti_residual = nib.nifti1.Nifti1Image(residuals.astype(np.float32), self.data.get_affine())




        if os.path.exists(self.masked_dwi):
            img = nib.load(self.masked_dwi)
            data = img.get_data()

            bvals, bvecs = read_bvals_bvecs(self.bval, self.bvec)
            gtab = gradient_table(bvals, bvecs)

            values = np.array(bvals)
            ii = np.where(values == bvals.min())[0]
            b0_average = np.mean(data[:,:,:,ii], axis=3)

            dkimodel = dki.DiffusionKurtosisModel(gtab)
            dkifit = dkimodel.fit(data)

            estimate_data = dkimodel.predict(gtab, S0=b0_average)
            residuals = np.absolute(data - estimate_data)

            Kfit_img = nib.Nifti1Image(dkifit.astype(np.float32), img.get_affine())
            res_img = nib.Nifti1Image(residuals.astype(np.float32), img.get_affine())
            nib.save(Kfit_img, self.kurtosis_img)
            nib.save(res_img, self.dki_res_img)

# class WholeImage(Fit):
#
#     def __init__(self, data, mask, gradient_table, fit_method, out_dti=None, out_dti_residual=None, out_dki=None, out_dki_residual=None, out_dti_fa=None, out_dti_md=None, out_dti_rd=None, out_dti_ad=None, out_dki_fa=None, out_dki_md=None, out_dki_rd=None, out_dki_ad=None):
#         Fit.__init__(self, data, mask, gradient_table, fit_method, out_dti, out_dti_residual, out_dki, out_dki_residual, out_dti_fa, out_dti_md, out_dti_rd, out_dti_ad, out_dki_fa, out_dki_md, out_dki_rd, out_dki_ad)
#
#     def createMask(self):
#         if os.path.exists(self.raw_data):
#             img = nib.load(self.raw_data)
#             data = img.get_data()
#             masked_data, mask = median_otsu(data, 2,2)
#
#             #Save those files
#             masked_img = nib.Nifti1Image(masked_data.astype(np.float32), img.get_affine())
#             mask_img = nib.Nifit1Image(mask.astype(np.float32), img.get_affine())
#
#             nib.save(masked_img, self.masked_dwi)
#             nib.save(mask_img, self.mask)


class SliceFit(Fit):

    def __init__(self, data, mask, gradient_table, fit_method, out_dti=None, out_dti_residual=None, out_dki=None, out_dki_residual=None, out_dti_fa=None, out_dti_md=None, out_dti_rd=None, out_dti_ad=None, out_dki_fa=None, out_dki_md=None, out_dki_rd=None, out_dki_ad=None):
        Fit.__init__(self, data, mask, gradient_table, fit_method, out_dti, out_dti_residual, out_dki, out_dki_residual, out_dti_fa, out_dti_md, out_dti_rd, out_dti_ad, out_dki_fa, out_dki_md, out_dki_rd, out_dki_ad)
        self.raw_data = slice_data

    def createMask(self):
        if os.path.exists(self.raw_data):
            img = nib.load(self.raw_data)
            data = img.get_data()
            masked_data, mask = median_otsu(data, 2,2)

            #Save those files
            masked_img = nib.Nifti1Image(masked_data.astype(np.float32), img.get_affine())
            mask_img = nib.Nifit1Image(mask.astype(np.float32), img.get_affine())

            nib.save(masked_img, self.masked_dwi)
            nib.save(mask_img, self.mask)

#============================================================================
#             Run
#============================================================================

def run(rawargs):
    arguments = docopt(doc, argv=rawargs, version='Fit v{0}'.format(Version))

    configuration = {}

    #Try to load the image data. If successful, save it to configuration as "data"
    try:
        configuration["data"] = nib.load(arguments["--image"])
    except:
        print("The image you specified does not exist, or cannot be read.")
        sys.exit(1)

    #Try to load the mask data. If successful, save it to configuration as "mask"
    if arguments["--mask"] != None and arguments["--mask"] != "None":
        try:
            configuration["mask"] = nib.load(arguments["--mask"])
        except:
            print("The mask image you specified does not exist, or cannot be read.")
            sys.exit(1)
    else:
        configuration["mask"] = None

    #Try to load the bvec, bvals files. If successful, save it to configuration as "gradient_table"
    try:
        bvals, bvecs = read_bvals_bvecs(arguments["--bvals"], arguments["--bvecs"])
        configuration["gradient_table"] = gradient_table(bvals, bvecs)
    except:
        print("Could not read bvec and/or bval file")
        sys.exit(1)

    #Update configuration with more basic settings
    lookup = {"--out_dti": "out_dti",
              "--out_dti_residual": "out_dti_residual",
              "--out_dki": "out_dki",
              "--out_dki_residual": "out_dki_residual",
              "--out_dti_fa": "out_dti_fa",
              "--out_dti_md": "out_dti_md",
              "--out_dti_rd": "out_dti_rd",
              "--out_dti_ad": "out_dti_ad",
              "--out_dki_fa": "out_dki_fa",
              "--out_dki_md": "out_dki_md",
              "--out_dki_rd": "out_dki_rd",
              "--out_dki_ad": "out_dki_ad"}

    for key, value in lookup.iteritems():
        if arguments[key] == "None" or arguments[key] == None:
            configuration[value] = None
        else:
            configuration[value] = arguments[key]


    if arguments["--fit_method"].upper() in ["WLS", "LS", "NLLS", "RT"]:
        configuration["fit_method"] = arguments["--fit_method"].upper()
    else:
        print("'{0}' is not a valid fit method. Choose either 'WLS', 'LS', 'NLLS', or 'RT'".format(arguments["--fit_method"].upper()))
        sys.exit(1)

    #Delete this when debugging is finished.
    print(configuration)

    #Check to see if the user specified output files derived from DTI fitting. If so, proceed with DTI fitting.
    if arguments["--slice"] == None or arguments["--slice"] == "None":
        fitter = Fit(**configuration)

    else:
        configuration["slice"] = int(arguments["--slice"])
        fitter = SliceFit(**configuration)
    print(type(fitter))
    fitter.apply_mask()

    if len([key for key in configuration.keys() if "out_dti" in key and configuration[key] != None]) != 0:
        fitter.fit_dti()
    if len([key for key in configuration.keys() if "out_dki" in key and configuration[key] != None]) != 0:
        fitter.fit_dki()

    fitter.save()




#    #retrieves arguments
#    arguments = docopt(doc, argv=rawargs, version='Orientation Check v{0}'.format(Version))
#    #print(arguments)
#    inputs = [{"Value":"image file", "Flag": "--image"}, {"Value":"bvec file", "Flag": "--bvec"}, {"Value":"bvec file", "Flag": "--bvec"}]
#    if arguments["--image_mask"] != None and arguments["--image_mask"] != 'None':
#        inputs.append({"Value":"image mask file", "Flag": "--image_mask"})
#    for inputinfo in inputs:
#        if not exists(arguments[inputinfo["Flag"]]):
#            print("The {0} specified does not exist!".format(inputinfo["Value"]))
#            sys.exit(1)
#
#
#        #Generate average signal, to be used in reverse calculation
#        image_average_signal = np.mean(image_masked, axis=3)
#
#        #Define the paths to the output files
#        spd_file_path = arguments["--outprefix"]+'_spd.nii.gz'
#        estimate_file_path = arguments["--outprefix"]+'_ecc_estimated.nii.gz'
#        error_file_path = arguments["--outprefix"]+'_ecc_error.nii.gz'
#
#        #Generate the SPD data (lower triangular of the symmetric matrix)
#        spd_data = result.lower_triangular()
#        #Predict the original data based on the resulting tensor data
#        estimate_data = result.predict(gtab, S0=image_b0_average_masked)
#        #Generate the difference between original and the predicted original
#        error_data = np.absolute(image_masked - estimate_data)
#
#    #     spd_file_path = arguments["--outprefix"]+arguments["--axis"]+pad_val(arguments["--slice"])+'_spd.nii.gz'
#    #     spd_data = result.lower_triangular()
#    #     estimate_file_path = arguments["--outprefix"]+arguments["--axis"]+pad_val(arguments["--slice"])+'_ecc_estimated.nii.gz'
#    #     #THE FOLLOWING DOES NOT WORK: image_mask is 3D, and result.predict(gtab) produces a 4D dataset. Will need to mask differently
#    #     estimate_data = result.predict(gtab, S0=image_average_signal_slice)# * image_mask
#    #     error_file_path = arguments["--outprefix"]+arguments["--axis"]+pad_val(arguments["--slice"])+'_ecc_error.nii.gz'
#    #     error_data = numpy.absolute(image_masked[imagedict[arguments["--axis"]]["scope"]] - estimate_data)
#
#    print("Saving SPD image to "+spd_file_path)
#    nib.save(nib.Nifti1Image(spd_data, image.get_affine()), spd_file_path)
#    print("Saving estimated image to "+estimate_file_path)
#    nib.save(nib.Nifti1Image(estimate_data, image.get_affine()), estimate_file_path)
#    print("Saving error image to "+error_file_path)
#    nib.save(nib.Nifti1Image(error_data, image.get_affine()), error_file_path)
#
#    #nib.save(nib.Nifti1Image(result.lower_triangular(), image.get_affine()), arguments["--outprefix"]+'_spd.nii.gz')
#
#
#
#    sys.exit(0)
#

#============================================================================
#             Main
#============================================================================

if __name__ == '__main__':
   args = sys.argv
   del args[0]
   run(args)
