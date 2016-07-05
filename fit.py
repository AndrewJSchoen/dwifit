#def run(rawargs):
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
#if __name__ == '__main__':
#    args = sys.argv
#    del args[0]
#    run(args)
#
#
#    ############################################################################

#!../Python/bin/python
import matplotlib.pyplot as plt
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
    fit.py [options] --bval=<FILE> --bvec=<FILE> --outprefix=<FILE> --fit_type=<type> --data_type=<type> --image=<FILE>

Options:
    -h --help            Show this screen.
    -v --version         Show version.
    --bval=<FILE>        BVAL File (path).
    --bvec=<FILE>        BVEC File (path).
    --outprefix=<FILE>   Output File (path).
    --fit_type=<type>    Specify a type (WLS, LS, NLLS, RT) [default: WLS].
    --data_type=<type>   Specify type of data (Whole_Image, Slice) [default: Whole_Image].
    --image=<FILE>       Image File (path).
""".format(Version)

#============================================================================
#             General Utility
#============================================================================

class Fit(object):

    def __init__(self, bvals, bvecs, outprefix, fit_method='WLS', data_type='Whole_Image', dwi_data):
        self.bval = bvals
        self.bvec = bvecs
        self.output_root_path = outprefix
        self.fit_method = fit_method
        self.raw_data = dwi_data

        #set up directories for storage of information and data
        self.output_data = self.output_root_path + "/data" #directory where all files are stored
        self.raw_data = self.output_data + "/dwi.nii.gz"
        self.mask = self.output_data + "/mask.nii.gz"
        self.masked_dwi = self.output_data + "/dwi.mask.nii.gz"
        self.bval = self.output_data + "/bvals"
        self.bvec = self.output_data + "/bvecs"
        self.tenfit_img = self.output_data + "/dti.tensor.nii.gz"
        self.dti_res_img = self.output_data + "/dti.res.nii.gz"
        self.kurtosis_img = self.output_data + "/dki.model.nii.gz"
        self.dki_res_img = self.output_data + "/dki.res.nii.gz"

    def setupDirectories(self):

        if not os.path.exists(self.output_root_path):
            os.mkdir(self.output_root_path)

        if not os.path.exists(self.output_preprocess):
            os.mkdir(self.output_preprocess)


    def TensorFit(self):
        if os.path.exists(self.masked_dwi):
            img = nib.load(self.masked_dwi)
            data = img.get_data()

            bvals, bvecs = read_bvals_bvecs(self.bval, self.bvec)
            gtab = gradient_table(bvals, bvecs)

            values = np.array(bvals)
            ii = np.where(values == bvals.min())[0]
            b0_average = np.mean(data[:,:,:,ii], axis=3)

            tenmodel = dti.TensorModel(gtab, fit_method="fit_method")
            tenfit = tenmodel.fit(data)

            spd_data = tenfit.lower_triangular()
            nib.save(nib.Nifti1Image(spd_data, image.get_affine()), spd_file_path)

            estimate_data = tenmodel.predict(gtab, S0=b0_average)
            residuals = np.absolute(data - estimate_data)

            fit_img = nib.Nifti1Image(tenfit.astype(np.float32), img.get_affine())
            res_img = nib.Nifti1Image(residuals.astype(np.float32), img.get_affine())
            nib.save(fit_img, self.tenfit_img)
            nib.save(res_img, self.dti_res_img)

    def KurtosisModelFit(self):
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

class WholeImage(Fit):

    def __init__(self, rawDWI):
        Fit.__init__(self, bvals, bvecs, outprefix, fit_method='WLS', "Whole_Image")
        self.raw_data = rawDwi

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


class SliceImage(Fit):

    def __init__(self, slice_data):
        Fit.__init__(self, bvals, bvecs, outprefix, fit_method='WLS', "Slice")
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
