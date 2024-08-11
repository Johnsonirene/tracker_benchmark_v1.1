% This function implements the VACF tracker.
function [results] = tracker(params)

frame_interval = params.frame_interval;
alphas = params.alphas;
update_interval=2;
num_frames     = params.no_fram;
newton_iterations = params.newton_iterations;
global_feat_params = params.t_global;
featureRatio = params.t_global.cell_size;
search_area = prod(params.wsize * params.search_area_scale);
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);
learning_rate = params.learning_rate;

[currentScaleFactor, base_target_sz, ~, sz, use_sz] = init_size(params,target_sz,search_area);
[y, cos_window] = init_gauss_win(params, base_target_sz, featureRatio, use_sz);
yf          = fft2(y);
[features, im, colorImage] = init_features(params);
[ysf, scale_window, scaleFactors, scale_model_sz, min_scale_factor, max_scale_factor] = init_scale(params,target_sz,sz,base_target_sz,im);

% Pre-computes the grid that is used for score optimization
ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
small_filter_sz = floor(base_target_sz/featureRatio);

% reg_window = construct_regwindow(params, use_sz, small_filter_sz);

time = 0;
loop_frame = 1;
Vy=0;
Vx=0;
% avg_list=zeros(num_frames,1);
% avg_list(1)=0;

for frame = 1:num_frames
    im = load_image(params, frame, colorImage);
    tic();  
    %% main loop
    
    if frame > 1
        pos_pre = pos;

        [xtf, xcf_c, pos, translation_vec, ~, ~, ~] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,wf,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame);
        Vy = pos(1) - pos_pre(1);
        Vx = pos(2) - pos_pre(2);
               
        % search for the scale of object
        [xs,currentScaleFactor,recovered_scale]  = search_scale(sf_num,sf_den,im,pos,base_target_sz,currentScaleFactor,scaleFactors,scale_window,scale_model_sz,min_scale_factor,max_scale_factor,params);
    end
    % update the target_sz via currentScaleFactor
    target_sz = round(base_target_sz * currentScaleFactor);
    %save position
    rect_position(loop_frame,:) = [pos([2,1]) - (target_sz([2,1]))/2, target_sz([2,1])];
    
    if frame == 1 

        reg_window = construct_regwindow(params, use_sz, small_filter_sz);
        s = reg_window;
        model_s = reg_window;
%         s_p = model_s;
%         s_pp = model_s;
        % extract training sample image region
        pixels = get_pixels(im, pos, round(sz*currentScaleFactor), sz);

        % 设置去噪参数
        edist = sqrt(sum(pixels.^2,3));
        patchSigma = sqrt(var(edist(:)));
        degreeOfSmoothing = 0.04  * patchSigma; % 根据图像的噪声水平调整
        searchWindowSize = 11;  % 搜索窗口大小
        comparisonWindowSize = 5;  % 比较窗口大小
        
        % 应用非局部均值去噪
        denoised_img = imnlmfilt(pixels, 'DegreeOfSmoothing', degreeOfSmoothing, ...
                                            'SearchWindowSize', searchWindowSize, ...
                                            'ComparisonWindowSize', comparisonWindowSize);

        context_m = context_mask(denoised_img,round(target_sz/currentScaleFactor));
        x = get_features(denoised_img, features, params.t_global);
        ct_m = mexResize(context_m,[size(x,1) size(x,2)],'auto');
        xc = x .* ct_m;
        xf=fft2(bsxfun(@times, x, cos_window));
        xcf_c=fft2(bsxfun(@times, xc, cos_window));
        xcf_p = zeros(size(xcf_c));
        model_xf = xf;
        model_xf_p = bsxfun(@times, zeros(size(xf), 'single'), xf);
        wf_p = bsxfun(@times, zeros(size(xf), 'single'), xf);
        wf_pp = bsxfun(@times, zeros(size(xf), 'single'), xf);

    elseif (frame < frame_interval)
        wf_pp = bsxfun(@times, zeros(size(xf), 'single'), xf);
        s_set{3} = s;
%         s_p = alphas  * s + (1 - alphas) * s_p;
%         model_s = 2 * s_p - s_pp;

    else
        % use detection features
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
        xcf_c = shift_sample(xcf_c, shift_samp_pos, kx', ky');
        model_xf_p = model_xf_set{frame_interval - 1};
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
        wf_p = wf_set{frame_interval - 1};
        wf_pp = wf_set{frame_interval - 2};
        s = s_set{frame_interval};
        s_p = s_set{frame_interval - 1};
        s_pp = s_set{frame_interval - 2};
        model_s = 2 * s_p - s_pp;
        model_s = alphas  * s + (1 - alphas) * model_s;

    end
    
    % context residual
    xcf = xcf_c - xcf_p;
    
%     [wf, hf] = run_training(model_xf, xcf,  model_xf_p, wf_p, use_sz, params, yf, small_filter_sz, s);
%     model_s = alphas * s + (1-alphas) * model_s;
%     s = admm_solve_s(use_sz, params, model_s, hf);
    
   % ADMM solution for scale estimation
   for iteration = 1 : params.al_iteration - 1
       [wf, hf, Sxy] = run_training(model_xf, xcf, model_xf_p, wf_p, wf_pp, use_sz, params, yf, small_filter_sz, s, model_s);
        s = admm_solve_s(params, use_sz, model_s, hf);
   end
    
    xcf_p = xcf_c;

    % Save xf and wf
    if frame <= frame_interval
        model_xf_set{frame} = model_xf;
        wf_set{frame} = wf;
        s_set{frame} = s;
    else
        model_xf_set(1:frame_interval-1) = model_xf_set(2:frame_interval);
        model_xf_set{frame_interval} = model_xf;
        wf_set(1:frame_interval-1) = wf_set(2:frame_interval);
        wf_set{frame_interval} = wf;
        s_set(1:frame_interval-1) = s_set(2:frame_interval);
        s_set{frame_interval} = s;
    end
    
    %% Update Scale
    if frame==1
%         xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz, 0);
        xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    else
        xs= shift_sample_scale(im, pos, base_target_sz,xs,recovered_scale,currentScaleFactor*scaleFactors,scale_window,scale_model_sz);
    end
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end

    time = time + toc();

     %%   visualization
    if params.visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        figure(1);
        imshow(im);
        if frame == 1
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 26, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
        else
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 28, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            text(12, 66, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
         end
        drawnow
    end
    loop_frame = loop_frame + 1;

%   save resutls.
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
% results.wf = wf;
% results.hf = hf;
% results.Sxy = Sxy;
% results.model_xf = model_xf;
end
%   show speed
disp(['fps: ' num2str(results.fps)])
