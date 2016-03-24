clear variables;
close all;
clc;

setenv('LD_RUN_PATH', [pwd, '/../../lib']);

mex -compatibleArrayDims clm_expr2.cpp ../../lib/librelaxometrycore.so
copyfile('clm_expr2.mexa64', '../../matlab/clm_expr2', 'f');

mex -compatibleArrayDims clmf_expr2.cpp ../../lib/librelaxometrycore.so
copyfile('clmf_expr2.mexa64', '../../matlab/clmf_expr2', 'f');

mex -compatibleArrayDims cudalm_expr2.cpp ../../lib/librelaxometrycore.so
copyfile('cudalm_expr2.mexa64', '../../matlab/cudalm_expr2', 'f');

mex -compatibleArrayDims cudalmf_expr2.cpp ../../lib/librelaxometrycore.so
copyfile('cudalmf_expr2.mexa64', '../../matlab/cudalmf_expr2', 'f');