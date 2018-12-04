% % creating a toy example -
% medium of 1 voxel, constant density

path = what('../');
path = path.path;
% change density (range [0, 1]) to create the desired scene 
density = 0.9;

% size of medium - in this case one voxel
W = 10;
L = 10;
H = 2;

data = density * ones(W, L, H);
volfilename = ['10VoxelMedium_beta', num2str(density), '.vol'];
res = [1, 1, 1];

data2vol(data, volfilename, path, res);