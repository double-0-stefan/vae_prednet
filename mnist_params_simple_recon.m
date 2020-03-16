function mnist_parameters_movie_recons(DEM,c)
% creates a movie of visual prusuit in extrinsic and intrinsic coordinates
% FORMAT spm_dem_pursuit_movie(DEM)
%
% DEM - DEM structure from reaching simulations
%
% hidden causes and states
%--------------------------------------------------------------------------
% x    - hidden states:
%   x.o(1) - oculomotor angle
%   x.o(2) - oculomotor angle
%   x.x(1) - target location (visual) - extrinsic coordinates (Cartesian)
%   x.x(2) - target location (visual) - extrinsic coordinates (Cartesian)
%   x.a(:) - attractor (SHC) states
%
% v    - causal states
%   v(1) - not used
%
% g    - sensations:
%   g(1) - oculomotor angle (proprioception)
%   g(2) - oculomotor angle (proprioception)
%   g(3) - target location (visual) - intrinsic coordinates (polar)
%   g(4) - target location (visual) - intrinsic coordinates (polar)
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston
% $Id: spm_dem_pursuit_movie.m 4625 2012-01-24 20:53:10Z karl $


% check subplot specifier
%--------------------------------------------------------------------------
try, c; catch, c = 0; end

clf, global stim ; global encoder, global real2model, global fov_width ; global real_digit
m  =  getEncoder() ;

% movie
%--------------------------------------------------------------------------
N      = length(DEM);

f = figure;
for i = 1:N

    for j = 1:length(DEM{i}.pPlots)
        subplot(1,2,1)
        
        cont_z = DEM{i}.pPlots{j};
        
        disc_z = full(sparse(real_digit,1,1,10,1));
        
        latents = spm_vec(cont_z,disc_z);
        prediction  = py_decode(latents);    
        py_out = m.model(0, prediction, pyargs('actions', [0 0], 'to_matlab', true));
        z_image = double(squeeze(py_out{2}.single));
        imagesc(z_image)      
        set(gca,'visible','off')
        MxHat(i) = getframe(gca);

        % sensory input
        %================F======================================================
        subplot(1,2,2)
    %     real_vis = reshape(DEM{i}.Y(:,end), 32, 32);
        imshow(strcat(int2str(real_digit), '.png'))
    %     imagesc(real_vis)
        set(gca,'visible','off')
        Mx(i) = getframe(gca);

        print(f, strcat('fig_',int2str(real_digit),'_',int2str(i),'_',int2str(j)), '-dpng');
    end 
end
  