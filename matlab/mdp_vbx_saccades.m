close all
clear all

% activate conda environment if applicable
conda.setenv('pytorch_cuda'); 

% if you edit a python file between runs, specify that module here to
% reload it. The alternative is to restart matlab.
mod = py.importlib.import_module('utils.dist');
py.importlib.reload(mod);
mod = py.importlib.import_module('modules');
py.importlib.reload(mod);
mod = py.importlib.import_module('load_encoder');
py.importlib.reload(mod);
mod = py.importlib.import_module('utils.model_utils');
py.importlib.reload(mod);
mod = py.importlib.import_module('model');
py.importlib.reload(mod);

exp_name = 'saccades';
model_name = 'l1_mnist_pnt1_vae1_normal_z20_h256_c50000_b2_elboTrue.pth';


% Load pytorch encoder (z | image) and set global 
saccade_encoder_path = fullfile('D:\', 'animal_ai', 'exps', exp_name, 'models', 'obs_model', model_name);
enc  = struct(get_model(saccade_encoder_path));
setEncoder(enc);
params = struct(enc.params);

% Load image to be encoded and set global
real_digit = 1;
digit_path = strcat(string(real_digit),'.png');
setStimulus(digit_path);

% Hacky convention for handling various python > matlab types
layers = params.layers / 1;
nT     = params.n_steps / 1;
zdim   = params.z_dim{1} / 1;

n_zc = int64(params.nz_con{1});
n_zd = int64(params.nz_dis{1}{1});

cont_ind = 1  : n_zc;
disc_ind = n_zc+1:zdim;

ldim   = cell(params.ldim{1});
ldim   = cellfun(@int64,ldim);
vector_ldim = prod(ldim);

% true causes (U) and priors (C) for every combination of discrete states
%--------------------------------------------------------------------------
N        = 16;                  % length of data sequence
n_digits = 10;
nh       = n_digits;            % number of hypotheses
nl        = vector_ldim / 2^4;   % number of locations

for i = 1:nh
    for j = 1:nl
    
        [xind, yind] = ind2sub([sqrt(nl) sqrt(nl)],j);
        o_action     = interp1([1,sqrt(nl)],[-1,1],[xind yind]);

        z            = py_encode(o_action);
        
        u = [ [ xind ; yind ] ; z(disc_ind)' ];

        demi.U{i,j} = u*ones(1,N);
        demi.C{i,j} = u*ones(1,N);
    end
end

% evaluate true and priors over causes given discrete states
%--------------------------------------------------------------------------
o     = [real_digit, nl/2];
O{1}  = spm_softmax(sparse(real_digit, 1,  1, nh, 1));
O{2}  = spm_softmax(sparse(nl/2, 1, 4, nl, 1));

% generative model
%==========================================================================
M(1).E.s = 1/2;                               % smoothness
M(1).E.n = 2;                                 % order of
M(1).E.d = 1;                                 % generalised motion

% hidden states
%--------------------------------------------------------------------------
x      = [0;0];                               % oculomotor angle
v.x    = [0;0];                               % fixed (attracting) point
v.h    = sparse(nh,1);                        % hypothesis

% level 1: Displacement dynamics and mapping to sensory/proprioception
%--------------------------------------------------------------------------
M(1).f  = @fx_dem;              % plant dynamics
M(1).g  = @gx_dem;              % prediction
M(1).x  = x;                                  % hidden states
M(1).V  = 2;                                   % error precision (g)
M(1).W  = 2;                                   % error precision (f)

% level 2:
%--------------------------------------------------------------------------
M(2).v = v;                                   % priors
M(2).V = [exp(1) exp(1) ones(1,nh)];

% generative model
%==========================================================================

% first level
%--------------------------------------------------------------------------
G(1).f  = @fx_adem;
G(1).g  = @gx_adem;
G(1).x  = [0;0];                              % hidden states
G(1).V = exp(1);                            % error precision
G(1).W = exp(1);                            % error precision
G(1).U = [1 1 zeros(1,nh)];                  % gain

% second level
%--------------------------------------------------------------------------
G(2).v = v;                                  % exogenous forces
G(2).a = [0;0];                              % action forces
G(2).V = exp(1);


% generate and invert
%==========================================================================
DEM.G = G;
DEM.M = M;

% solve and save saccade
%--------------------------------------------------------------------------
DEM    = mnist_MDP_DEM(DEM,demi,O,o);

nr    = 7;                                   % size of salience map
R     = sparse(nr*nr,1);                      % salience map with IOR


if ~exist('ADEM_saccades.mat','file')
    % (k) saccades
    %----------------------------------------------------------------------
    for k = 1:8
        k
        % solve and save saccade
        %------------------------------------------------------------------
        DEM     = spm_ADEM(DEM);
        DEM     = spm_ADEM_update(DEM);

        % overlay true values
        %------------------------------------------------------------------
        spm_DEM_qU(DEM.qU,DEM.pU)


        % store
        %------------------------------------------------------------------
        ADEM{k} = DEM;

    end
    save ADEM_saccades ADEM
end

load('ADEM_saccades')



% 
% save
%----------------------------------------------------------------------
%     

% create movie in extrinsic and intrinsic coordinates
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1');
mnist_search_plot(ADEM(1:end))
% 
% % create movie in extrinsic and intrinsic coordinates
% %--------------------------------------------------------------------------
% spm_figure('GetWin','Figure 2');
% spm_dem_search_trajectory(ADEM)
% 
% % create movie in extrinsic and intrinsic coordinates
% %--------------------------------------------------------------------------
% spm_figure('GetWin','Figure 3');
% STIM.R = spm_hanning(32)*spm_hanning(32)';
% spm_dem_search_movie(ADEM)
% 
% 
% M = spm_DEM_M_set(M);
% U    = ones(ldim,nT);
% DEM  = spm_DEM_generate(M,U);
% data = double(py.data_utils.get_dataset(m.params.dataset, m.params.b, pyargs('from_matlab', 1)));
% Y(:,:,1:nT) = permute(data(1:nT,:,:), [3,2,1]);
% DEM.C = Y;

%  second level (discrete: lexical)
%==========================================================================
% There are two outcome modalities (what and where), encoding one of 10
% digits and one of 64 locations.  The hidden states have four factors;
% corresponding to context (10 digits), where (the 64 locations)
% There are no specific prior
% preferences at this level, which means behaviour is purely epistemic.
%--------------------------------------------------------------------------
label.factor     = {'what','where'};
label.name{1}    = cellstr(string(1:nh));
label.name{2}    = cellstr(string(1:nl));


label.modality   = {'what','where'};
label.outcome{1} = cellstr(string(1:nl));
label.outcome{2} = cellstr(string(1:nl));


% prior beliefs about initial states
%--------------------------------------------------------------------------
clear D
D{1} = ones(nh,1);           % what:  digit
D{2} = zeros(nl,1); 
D{2}(sub2ind([sqrt(nl) sqrt(nl)],sqrt(nl)/2, sqrt(nl)/2)) = 1; % where:    {'1',...,'4'}

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
end
for f1 = 1:Ns(1)
    for f2 = 1:Ns(2)
        % A{1} what: digit
        % saccade to cue location
        %----------------------------------------------------------
        A{1}(f1,f1,f2) = 1;
        A{1}(f1,f1,f2) = 1;

        % A{2} where:
        %----------------------------------------------------------
        A{2}(f2,f1,f2) = 1;
    end
end

% controlled transitions: B{f} for each factor
%--------------------------------------------------------------------------
for f = 1:Nf
    B{f} = eye(Ns(f));
end

% controllable fixation points: move to the k-th location
%--------------------------------------------------------------------------
for k = 1:Ns(2)
    B{2}(:,:,k) = 0;
    B{2}(k,:,k) = 1;
end

% MDP structure for this level (and subordinate DEM level)
%--------------------------------------------------------------------------
mdp.T     = 6;                      % number of updates
mdp.A     = A;                      % observation model
mdp.B     = B;                      % transition probabilities
mdp.D     = D;                      % prior over initial states
mdp.DEM   = DEM;                    % same as ADEM{end}
mdp.demi  = demi;

mdp.label = label;
mdp.Aname = label.modality;
mdp.Bname = label.factor;
mdp.alpha = 64;
mdp.chi   = 1/32;

clear A B D

MDP= spm_MDP_check(mdp);

% invert this model of sentence reading
%==========================================================================
MDP  = spm_MDP_VB_X(MDP);

% show belief updates (and behaviour) over trials
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 3'); clf
spm_MDP_VB_trial(MDP);

function m = get_model(path)
    conda.setenv('pytorch_cuda')
    py.os.getcwd()
    mod = py.importlib.import_module('load_encoder');
    m = mod.get_model(path); 
end

% maybe add scaling coefficient 
function f = fx_dem(x, v, P)
    f = v.x - x;
end
function f = fx_adem(x, v, a, P) 
    f  = a - x;
end 

function g = gx_adem(x, v, a, P)
    m = getEncoder();
    s = getStimulus();
    a = a
    hm = m.model(full(s), pyargs('actions', a, 'to_matlab', true));
    fov = double(squeeze(hm{2}.single));
    g   = double(spm_vec(a,fov(:)));
end

function g = gx_dem(x, v, P)
    m = getEncoder();
    s = getStimulus();
    x = x
    hm = m.model(full(s), pyargs('actions', x, 'to_matlab', true));
    fov = double(squeeze(hm{2}.single));
    g   = double(spm_vec(x,fov(:)));
end

function setStimulus(num)
    global stim
    stim = imread(num);
    stim = stim(:,:,1);
    % nb we have resized mnist to 32x32
    stim = padarray(stim,[2 2],0,'both');

end


function setEncoder(val)
    global encoder
    encoder = val;
end


function m = getDecoder
    global decoder
    m = decoder;
end

function setDecoder(val)
    global decoder
    decoder = val;
end


function setRetina(val)
    global retina 
    retina = val;
end

function r = getRetina(val)
    global retina 
    r = retina;
end

