# What is it about?
The aim of this project is to speed up the calculation of relaxations maps in NMR. During the work on my bachelor's thesis at Graz Universtity of Technology the base was created. If you are interested in the thesis (completly written in German language) feel free to contact me.
# How does it work?
Based on a MATLAB implementation it is rewritten in C++ and NVIDIA CUDA. Levenberg-Marquardt algorithm and the extended version of it from Fletcher are used for solving the minimization problems. The speedup is reached through the parallelization on the gpu.
# Inputs
Supported input formats are DICOM and NIfTI. An adapter for MATLAB via mex function is also available.
# Outputs
R1/T1 maps using the monoexponential model. R2/T2/R2*/T2* maps using the monoexponential model. A more advanced model for T2 maps is also included. Goodness-of-Fit map calculation is also available to check the results.
# Usage
Example shows how to create a R2* map using the monoexponential model.
`relaxometry -p cudalmf -m expr2 -t 100 -- echotimes ./data/nii/input/T2star/1/echotimes.txt ../data/nii/input/T2star/1/gre6E.M.nii.gz R2star_cudalmf_expr2_1.nii.gz M0_cudalmf_expr2_1.nii.gz GoF_cudalmf_expr2_1.nii.gz`
# Dependencies
- [Grassroots DICOM implementation](https://sourceforge.net/projects/gdcm/)
- [nifticlib](https://sourceforge.net/projects/niftilib/files/nifticlib/)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone)
