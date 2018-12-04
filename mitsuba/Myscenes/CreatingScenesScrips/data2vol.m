function data2vol(data, volfilename, path, res)
% Generates a .vol file from volumetric data (3D matrix)
% Generates a .obj file: a bounding box for the volumetric data
% Inputs: data        - 3D matrix of double/single 
%         volfilename - output .vol file name  
%         path        - path of output files
%         res         - voxel resolusion (size)
% Code is based on .vol file description presented in [1]
%
% [1] -  http://www.mitsuba-renderer.org/releases/current/documentation.pdf
% chapter 8.7.2. Grid-based volume data source (gridvolume)

%% Write the header: Bytes 1water_gray-48 
volfilename = [path, '/', volfilename];

% rotated_data =  permute(data,[1 3 2]);
[N_x, N_y, N_z ,N_chan] = size(data);

%  Set resolution of the volume - [dx, dy, dz] [meteres]  
if exist('res','var') 
    if length(res)==1 
        dx = res ; dy = res ; dz = res ;
    elseif length(res)==3
        dx = res(1) ; dy = res(2); dz = res(3);
    end
else
    dx = 1/128; dy = 1/128 ; dz = 0.3906/50 ;
end

% Create bounding box according to botton-left & top-right corners
xmin = -N_x*dx*0.5; 
zmin = -N_z*dz*0.5;
ymin = -N_y*dy*0.5;
xmax = -xmin;
zmax = -zmin;
ymax = -ymin;
bounds_filename = [path, '/', 'bounds_padded.obj']; 
create_boundingbox_obj([xmin xmax],[ymin ymax],[zmin zmax],bounds_filename)
box = [xmin,ymin,zmin,xmax,ymax,zmax];

% Create .vol file with all relevant information
fid = fopen(volfilename,'w');
fwrite(fid,'VOL','char');    % Bytes 1-3 ASCII Bytes �V�, �O�, and �L�
fwrite(fid,3);               % Byte 4 File format version number (currently 3)
fwrite(fid,1,'uint32');      % Bytes 5-8 Encoding identifier (32-bit integer).The following choices are available:
                             %       1. Dense float32-based representation
                             %       2. Dense float16-based representation (currently not supported by this implementation)       
                             %       3. Dense uint8-based representation (The range 0..255 will be mapped to 0..1)
                             %       4. Dense quantized directions. The
                             %       directions are stored in spherical coordinates with a total storage cost of 16 bit per entry.
fwrite(fid,N_x,'uint32');   % Bytes 9-12 Number of cells along the X axis (32 bit integer)
fwrite(fid,N_y,'uint32');   % Bytes 13-16 Number of cells along the Y axis (32 bit integer)
fwrite(fid,N_z,'uint32');   % Bytes 17-20 Number of cells along the Z axis (32 bit integer)
fwrite(fid,N_chan,'uint32'); % Bytes 21-24 Number of channels (32 bit integer, supported values: 1 or 3)               
fwrite(fid,box,'single');    % Bytes 25-48 Axis-aligned bounding box of the data stored in single precision (order: xmin, ymin, zmin, xmax, ymax, zmax)

%% Write the data: Bytes 49-*
[buffer] = vol2Buffer(data);
fwrite(fid,buffer,'float32');
fclose(fid);

end


function create_boundingbox_obj(x,y,z,filename)
    % Inputs: x=[xmin xmax], y=[ymin ymax], z=[zmin zmax]
    %         filename - bounding box .obj file name    
    fid = fopen(filename,'w');
    % Define all the vertices 
    fprintf(fid,['v ' num2str(x(2)) ' ' num2str(y(2)) ' ' num2str(z(1)) '\n']); 
    fprintf(fid,['v ' num2str(x(2)) ' ' num2str(y(1)) ' ' num2str(z(1)) '\n']); 
    fprintf(fid,['v ' num2str(x(1)) ' ' num2str(y(1)) ' ' num2str(z(1)) '\n']); 
    fprintf(fid,['v ' num2str(x(1)) ' ' num2str(y(2)) ' ' num2str(z(1)) '\n']);
    fprintf(fid,['v ' num2str(x(2)) ' ' num2str(y(2)) ' ' num2str(z(2)) '\n']); 
    fprintf(fid,['v ' num2str(x(2)) ' ' num2str(y(1)) ' ' num2str(z(2)) '\n']);
    fprintf(fid,['v ' num2str(x(1)) ' ' num2str(y(1)) ' ' num2str(z(2)) '\n']); 
    fprintf(fid,['v ' num2str(x(1)) ' ' num2str(y(2)) ' ' num2str(z(2)) '\n']);
    % Define all the normals  
    fprintf(fid,'vn 0.000000 1.000000 0.000001\n');
    fprintf(fid,'vn 0.000000 1.000000 0.000000\n');
    fprintf(fid,'vn -1.000000 0.000000 -0.000000\n');
    fprintf(fid,'vn -0.000000 -1.000000 -0.000001\n');
    fprintf(fid,'vn -0.000000 -1.000000 0.000000\n');
    fprintf(fid,'vn 1.000000 0.000000 -0.000001\n');
    fprintf(fid,'vn 1.000000 -0.000001 0.000001\n');
    fprintf(fid,'vn -0.000000 -0.000000 1.000000\n');
    fprintf(fid,'vn 0.000000 0.000000 -1.000000\n');
    % Define the connectivity
    fprintf(fid,'f 5//1 1//1 4//1\n');
    fprintf(fid,'f 5//2 4//2 8//2\n');
    fprintf(fid,'f 3//3 7//3 8//3\n');
    fprintf(fid,'f 3//3 8//3 4//3\n');
    fprintf(fid,'f 2//4 6//4 3//4\n');
    fprintf(fid,'f 6//5 7//5 3//5\n');
    fprintf(fid,'f 1//6 5//6 2//6\n');
    fprintf(fid,'f 5//7 6//7 2//7\n');
    fprintf(fid,'f 5//8 8//8 6//8\n');
    fprintf(fid,'f 8//8 7//8 6//8\n');
    fprintf(fid,'f 1//9 2//9 3//9\n');
    fprintf(fid,'f 1//9 3//9 4//9\n');
    fclose(fid);
end


function [buffer] = vol2Buffer(data)
% Binary data of the volume stored in the specified encoding. The data are ordered so that the following C-style indexing operationmakes sense
% after the file has been mapped into memory: data[((zpos*N_y + ypos)*N_x + xpos)*channels + chan]
% where (xpos, ypos, zpos, chan) denotes the lookup location.
% Note - the formula is for 0-base indexing (C like)

    [N_x, N_y, N_z , channels] = size(data);
    buffer = zeros(numel(data),1);
    for xpos=1:N_x
        for ypos=1:N_y
            for zpos=1:N_z
                for chan=1:channels
                    pos = ((xpos-1) + ((ypos-1) + (zpos-1)*N_y)*N_x)*channels + (chan-1) + 1;
                    buffer(pos) = data(xpos,ypos,zpos,chan);
                end
            end
        end
    end
end
