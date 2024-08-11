function results = run_DSHCF(seq, rp, bSaveImage)

setup_paths;
%  Initialize path
addpath('feature/');
addpath('feature/lookup_tables/');
addpath('implementation/');
addpath('utils/');

%  HOG feature parameters
hog_params.cell_size = 4;
hog_params.nDim   = 31;

%  ColorName feature parameters
cn_params.nDim  =10;
cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;
ic_params.compressed_dim = 3;

%  Grayscale feature parameters
grayscale_params.nDim=1;
grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

% Which features to include
params.t_features = {
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_table_feature, 'fparams',ic_params),...
};

%  Global feature parameters
params.t_global.cell_size = 4;          % Feature cell size
params.t_global.normalize_power = 2;    % Lp normalization with this p
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
params.image_sample_size = 160^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

%   Gaussian response parameter
params.output_sigma_factor = 1/16;	 % standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.newton_iterations  = 5;           % number of Newton's iteration to maximize the detection scores

params.reg_window_min = 1e-3; % the minimum value of the regularization window
params.reg_window_max = 1e5;  % The maximum value of the regularization window

%  Set files and gt
params.name=seq.name;
params.video_path = seq.path;
params.img_files = seq.s_frames;
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram  = seq.endFrame - seq.startFrame + 1;
params.seq_st_frame = seq.startFrame;
params.seq_en_frame = seq.endFrame;
params.ground_truth = seq.init_rect;

%  Scale parameters
params.scale_sigma_factor = 0.51;   
params.num_scales = 33;
params.scale_step = 1.03;
params.scale_model_factor = 1.0;
params.scale_model_max_area = 32*16;
params.hog_scale_cell_size = 4;  
params.scale_lambda = 1e-4;      
params.learning_rate_scale = 0.025;

% TB-BiCF parameters
params.learning_rate = 0.0362;  % TB-BiCF online adaption rate 
params.t_lambda = 1.2;           % Regularization factor on TB-BiCF
params.t_lambda2 = 0.001;
params.gamma = 0.1;            % Regularization factor on bidirectional error
params.frame_interval = 3;     % Maximum frame interval within the temporary block

%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function.
params.admm_iterations = 2;
% params.admm_lambda = 0.01;
params.alphas = 1;
params.mu = 100;              % Initial penalty factor
params.beta = 50;             % Scale step
params.s_init = 7;
params.al_iteration = 3;

%   Debug and visualization
params.print_screen = 0;
params.visualization = 0;               % Visualiza tracking and detection scores
params.disp_fps = 1;

%   Run the main function
results = tracker(params);

end
