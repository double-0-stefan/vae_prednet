close all
clear all
clear global 
% ----------------------------------------------
%           Set up Python Environment 
% ----------------------------------------------
% Type 'pyversion' in matlab console to check the python env that your matlab is using
% Ensure pytorch is installed under this env, alternatively:

% activate conda environment (with pytorch installed)
% requires conda.m from https://uk.mathworks.com/matlabcentral/fileexchange/65873-condalab-control-and-view-anaconda-python-environments-form-within-matlab
conda.setenv('pytorch_cuda'); 

% If you edit a python file between matlab runs, specify that file here to
% reload it. The alternative is to restart matlab each time. 
mod = py.importlib.import_module('utils.dist');        py.importlib.reload(mod);
mod = py.importlib.import_module('modules');           py.importlib.reload(mod);
mod = py.importlib.import_module('load_encoder');      py.importlib.reload(mod);
mod = py.importlib.import_module('utils.model_utils'); py.importlib.reload(mod);
mod = py.importlib.import_module('model');             py.importlib.reload(mod);

% ------------------------------------------------
%               Pytorch model details
% ------------------------------------------------

% experiment subdirectory 
exp_name = 'saccades';
model_name = 'l1_mnist_pnt0_vae1_normal_z20_h512_c25000_b5_ld30_elboTrue_ela6.pth';   

% bijective (assuming sufficient disentanglemnt) map from real to model states
% i.e real2model(1) = index corresponding to the agent's representation of 1
% this needs to be derived by visual inspection of:
% exps/exp_name/plots/obs_model/model_name/traversal_grid_end.png for each model

% Width of the foveation window must be defined at the time of loading 
% a model. This can be defined elsewhere, but the model must be reloaded
% with the new foveation_windows size. see modules.py - Retina.__init__
global fov_width  ; fov_width = 32;

% Load pytorch model and set global 
saccade_encoder_path = fullfile('D:\', 'animal_ai', 'exps', exp_name, 'models', 'obs_model', model_name);
enc  = struct(get_model(saccade_encoder_path, py.int(fov_width)));
setEncoder(enc);
params = struct(enc.params);

% Get some parameters from model file 
zdim   = double(params.z_dim{1});   % n latent variables
n_zc   = double(params.nz_con{1});  % n continuous latent variables 
cont_ind = 1  : n_zc;               % & their idxs
n_zd = double(params.nz_dis{1}{1}); % n categorical latent variables 
disc_ind = n_zc+1:zdim;             % & their idxs

% ---------------------------------------------------------
%                   Matlab / DEM setup   
% ---------------------------------------------------------

global real_digit ;

T  = 10;

    
for digit = 1:10
    
    % Load stimulus and set global
    real_digit = digit;
    digit_path = strcat(string(real_digit),'.png');
    % Global is needed by generative process & plotting routines
    setStimulus(digit_path);

    nh    = n_zd;  % number of hypotheses
    
    prcdbg = 8;     % scale precision for debugging
    init_digit = full(sparse(real_digit,1,1,nh,1));

    % ----------------------------------------------------------
    %                   HDM Specification
    % ----------------------------------------------------------

    M(1).E.s = 1/2;                               % smoothness
    M(1).E.n = 1;                                 % order of
    M(1).E.d = 2;                                 % generalised motion

    % Initial states / causes
    prcdbg = 8;

    % v.h = = -log(nh)*ones(nh,1); % flat prior on digit identity

    % level 1: Displacement dynamics and mapping to sensory/proprioception
    M(1).f  = @fx_dem;           % dynamics ( see bottom of this script )
    M(1).g  = @gx_dem;           % prediction
    M(1).x  = 0;                 % hidden states
    M(1).W  = exp(prcdbg);       % error precision (f)
    M(1).V  = exp(prcdbg);   % error precision (g)
    % parameters to be learned
    M(1).pE = zeros(n_zd,1);
    % prior expectation of their covariance
    M(1).pC = 2.5;

    % level 2:
    %--------------------------------------------------------------------------
    M(2).v  = 0;                 % priors
    M(2).V  = exp(prcdbg * 2);   % error precision

    % first level
    %--------------------------------------------------------------------------
    G(1).f  = @fx_adem;
    G(1).g  = @gx_adem;
    G(1).x  = 0;

    % second level
    %--------------------------------------------------------------------------
    G(2).v  = 0; % must agree with demi.C dimensions
%     G(2).V  = 128;

    DEM.G  = G;
    DEM.M  = M;
    DEM.db = 0; % visualise (bool) 

    DEM.C = zeros(1,T);
    DEM.U = zeros(1,T);
    
    p = zeros(10, T);
    c = zeros(10, T);
    ADEMS = {};
    % let initial guess be displayed for a while
    for t = 1:8
        DEM.pPlots{t} = zeros(10,1);    
    end

    ADEMS{1} = DEM;
    for t = 1:T

        DEM    = spm_ADEM(DEM);
        p(:,t) = spm_vec(DEM.M(1).pE);
        c(:,t) = spm_vec(diag(DEM.M(1).pC));   
        DEM    = spm_ADEM_update(DEM);    
        ADEMS{t+1} = DEM;
    end

    mnist_params_simple_recon(ADEMS);

    clear ADEMS DEM M G
    close all
end


for digit = 1:10

    vname = strcat('vid_', int2str(digit), '.avi');
    writerObj = VideoWriter(vname);
    writerObj.FrameRate = 10;
    open(writerObj)

    for i = 1:11
        for j = 1:8
            frame = imread(strcat('fig_',int2str(digit), '_', int2str(i),'_',int2str(j), '.png'));
            frame = frame(:,120:end-100,:);
            writeVideo(writerObj,frame); %write the image to file
        end
    end
    close(writerObj); %close the file
end

function g = gx_dem(x, v, P)
    global real_digit;
    vh = full(sparse(real_digit,1,1,10,1));
    latents  = spm_vec(P, vh);
    g = py_foveate(py_decode(latents), [0 0]) ;  
    g   = double( g(:) );

end

function g = gx_adem(x, v, a, P)
    s = getStimulus();
    s = py_foveate(s, [0 0]);
    g   = double( s(:) );
end

function f = fx_adem(x,v,a,P)
    f = x;
end

function f = fx_dem(x,v,P)
    f = x;
end

function setStimulus(num)
    global stim
    stim = imread(num);
    if length(size(stim)) > 2
       stim = rgb2gray(stim);
    end
    stim = im2double(stim);
    stim = padarray(stim,[2 2],0,'both');

end

function m = get_model(path, g, noise)
    conda.setenv('pytorch_cuda')
    py.os.getcwd()
    mod = py.importlib.import_module('load_encoder');
    switch nargin
        case 1
            m = mod.get_model(path); 
        case 2
            m = mod.get_model(path, g);
        case 3
            m = mod.get_model(path, g, noise);
    end
end

function setEncoder(val)
    global encoder
    encoder = val;
end
