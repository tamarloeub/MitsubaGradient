% % creating a toy example -
% medium of 1 voxel, constant density
clear;
close all;
clc

path = what('../');
path = path.path;
% change density (range [0, 1]) to create the desired scene 
density = 1;

% size of medium - in this case one voxel
W = 2;
L = 2;
H = 2;

% data = density * ones(W, L, H);
data = linspace(1, W * L*  H, W * L*  H);
data = reshape(data, [W, L, H]);
volfilename = ['mixMedium_beta', num2str(density), '.vol'];
res = [1, 1, 1];

data2vol(data, volfilename, path, res);