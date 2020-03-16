function spm_dem_search_plot(DEM)
% plots visual search in extrinsic and intrinsic coordinates
% FORMAT spm_dem_search_plot(DEM)
%
% DEM - {DEM} structures from visual search simulations
%
% hidden causes and states
%==========================================================================
% x    - hidden states:
%   o(1) - oculomotor angle
%   o(2) - oculomotor angle
%   x(1) - relative amplitude of visual hypothesis 1
%   x(2) - relative amplitude of visual hypothesis 2
%   x(3) - ...
%
% v    - hidden causes
%
% g    - sensations:
%   g(1) - oculomotor angle (proprioception - x)
%   g(2) - oculomotor angle (proprioception - y)
%   g(3) - retinal input - channel 1
%   g(4) - retinal input - channel 2
%   g(5) - ...
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston
% $Id: spm_dem_search_plot.m 4851 2012-08-20 15:03:48Z karl $
 
 
% Preliminaries
%--------------------------------------------------------------------------
clf, global stim, global encoder 
N  = length(DEM);
S  = stim;
 
% Stimulus
%======================================================================
Dx = 32/2;
Dy = 32/2;
a  = [];
q  = [];
c  = [];
 
for i = 1:N
    
    % i-th saccade - position
    %----------------------------------------------------------------------
    pU = DEM{i}.pU.x{1}(1:2,:)*16;
    qU = DEM{i}.qU.x{1}(1:2,:)*16;
    T  = length(pU);
    
    % conditional confidence
    %----------------------------------------------------------------------
    qC = DEM{i}.qU.S;
    
    for t = 1:length(qC)
        qV(t) = 1;
    end
    
    % accumulate responses
    %----------------------------------------------------------------------
    a  = [a DEM{i}.qU.a{2}];                % action
    q  = [q DEM{i}.qU.x{1}(3:end,:)];       % hidden perceptual states
    c  = [c qV];                            % conditional variance
    
    
    % eye movements in extrinsic coordinates
    %======================================================================
    subplot(6,N,i)
    
    image((S + 1)*32), axis image off, hold on
    plot(qU(2,T) + Dy,qU(1,T) + Dx,'.g','Markersize',8)
    plot(pU(2,T) + Dy,pU(1,T) + Dx,'.r','Markersize',16)
    drawnow, hold off
    
    o   = full(DEM{i}.pU.x{1}(:,T));
    py_sample = encoder.model(stim, pyargs('actions', o, 'to_matlab', true));    
    
    z       = double(py_sample{1}.single);
    z_image = double(py_sample{2}.single);
    
    
%     salience map
%     ======================================================================
    subplot(6,N,i + N*1)
%     DEM{i}.qU
    prediction = py_decode(z);
    imagesc(prediction), axis image off
    
    
    % sensory input
    %================F======================================================
    subplot(6,N,i + N*3)
    
    % i-th saccade - sensory samples
    %----------------------------------------------------------------------

    s_size = size(z_image);
    s = reshape(z_image,[s_size(end),s_size(end-1)]);
    
    imagesc(s), axis image off
    
    
    % percept
    %======================================================================
    subplot(6,N,i + N*5)
    
%     % i-th saccade - percept
%     %----------------------------------------------------------------------
%     qU    = DEM{i}.qU.x{1}(3:end,:);
%     
%     % hypotheses (0 < H < 1 = normalised neg-entropy)
%     %----------------------------------------------------------------------
%     h     = spm_softmax(qU(:,T),2);
%     H     = 1 + h'*log(h)/log(length(h));
%     
%     % retinotopic predictions
%     %----------------------------------------------------------------------
%     s     = 0;
%     for j = 1:length(h)
%         s = s + h(j)*spm_read_vols(S{j});
%     end
%     image(s*H*64), axis image off
    
end
 
% set ButtonDownFcn
%--------------------------------------------------------------------------
t  = (1:length(a))*12;
subplot(6,1,3)
plot(t,a')
title('Action (EOG)','FontSize',16)
xlabel('time (ms)')
axis([1 t(end) -1 1])
 
subplot(6,1,5)
% spm_plot_ci(q(1,:),c,t); hold on
bar(spm_softmax(DEM{1}.qU.v{1}(2+7*7+1:end,end)))
% plot(t,q), hold off
% axis([1 t(end) -8 8])
title('Posterior belief','FontSize',16)
xlabel('time (ms)')
