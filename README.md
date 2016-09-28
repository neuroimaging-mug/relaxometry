# MRI-fitting: Fast estimation of relaxometry times using CUDA
This work will be presented at the 33rd annual meeting of the ESMRMB 2016 in Vienna

Presentation number: 531

Abstract ID: 687

29.09.2016, 14:00-15:00

30.09.2016, 10:50-11:50

Desk Nr: 4

# Purpose of the software
The here presented relaxometry tool was designed to speed up voxelwise fits of monoexponential functions with two or three parameters to estimate T1, T2 or T2* relaxation times and was optimized for the analysis of large clinical cohorts. As a central asset, the software uses the power of the graphics-processing unit (GPU) which enables a significant decrease in computing time and which is perfectly suited for the parallelization of such fits. A special feature of the presented tool is its modular structure, which allows the selection of different solvers for each model and which also facilitates the implementation of other functions for voxelwise fits. 

# Methods
The software was realized in C++ and CUDA 7.5 on an Arch Linux and Ubuntu 14.04 system. The implemented solver algorithms are a simple linear regression, Levenberg-Marquardt (LM) and an optimized Levenberg-Marquardt-Fletcher (LMF) algorithm [1,2]. For further acceleration, we first segmented the region of interests using the "bet" tool by FSL [3]. The input images can be in DICOM or NIfTI format.  Implementation is open to new solvers and models, so far an extended T2 model has already been included [4]. To test the algorithm, the following data were used: T2* mapping was performed with data from a 3D multi-echo gradient echo sequence with 64 slices, 6 echoes, resolution of 0.9x0.9x2mm and matrix-dimensions 208x256x64voxel. For T1 mapping we used an inversion recovery sequence (6 inversion times, resolution 2x2x4mm, matrix: 84x128x15voxel) as the gold standard for assessing T1 relaxation time. 

# Inputs
Supported input formats are DICOM and NIfTI. An adapter for MATLAB via mex function is also available.
# Outputs
R1/T1 maps using the monoexponential model. R2/T2/R2\*/T2\* maps using the monoexponential model. A more advanced model for T2 maps is also included. Goodness-of-Fit map calculation is also available to check the results.
# Usage
Example shows how to create a R2* map using the monoexponential model.

`relaxometry -p cudalmf -m expr2 -t 100 --echotimes ./data/nii/input/T2star/1/echotimes.txt ../data/nii/input/T2star/1/gre6E.M.nii.gz R2star_cudalmf_expr2_1.nii.gz M0_cudalmf_expr2_1.nii.gz GoF_cudalmf_expr2_1.nii.gz`
# Dependencies
- [Grassroots DICOM implementation](https://sourceforge.net/projects/gdcm/)
- [nifticlib](https://sourceforge.net/projects/niftilib/files/nifticlib/)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone)

# References
[1]   R. Fletcher (1971): A Modified Marquardt Subroutine for Nonlinear Least Squares; Harwell
[2]   M. Balda (2012): LMFnlsq - Solution of nonlinear least squares; http://de.mathworks.com/matlabcentral/fileexchange/17534-lmfnlsq-solution-of-nonlinear-least-squares
[3]   S.M. Smith (2012): Fast robust automated brain extraction. Human Brain Mapping, 17(3):143-155
[4]   A. Petrovic, E. Scheurer, R. Stollberger (2015): Closed-form solution for T2 mapping with nonideal refocusing of slice selective CPMG sequences; MRM

