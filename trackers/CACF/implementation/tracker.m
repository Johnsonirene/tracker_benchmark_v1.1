% This function implements the VACF tracker.
function [results] = tracker(params)

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
smallsz = floor(base_target_sz/featureRatio);
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
        [xtf, xcf_c, pos, translation_vec, ~, ~, ~] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame);
        Vy = pos(1) - pos_pre(1);
        Vx = pos(2) - pos_pre(2);
               
        % search for the scale of object
        [xs,currentScaleFactor,recovered_scale]  = search_scale(sf_num,sf_den,im,pos,base_target_sz,currentScaleFactor,scaleFactors,scale_window,scale_model_sz,min_scale_factor,max_scale_factor,params);
    end
    % update the target_sz via currentScaleFactor
    target_sz = round(base_target_sz * currentScaleFactor);
    %save position
    rect_position(loop_frame,:) = [pos([2,1]) - (target_sz([2,1]))/2, target_sz([2,1])];
                                    
    if frame==1 
        % extract training sample image region
        pixels = get_pixels(im, pos, round(sz*currentScaleFactor), sz);
        context_m = context_mask(pixels,round(target_sz/currentScaleFactor));
        x = get_features(pixels, features, params.t_global);         
        XX = reshape(x, [], size(x, 3));  
        XX=XX';
        data=XX;
        cluster_n=2;
        [cen, dist,U, obj_fcn] = FCMClust(data, cluster_n);
        % 初始化权重向量

% 对于每个特征通道，确定其属于哪个簇类
for i = 1:size(x,3)
    if U(1, i) > U(2, i)
        % 属于第一簇类
        if U(1, i) > 0.7
            weights(i) = 1;
        else
         diss = dist(1, i);
         max_1=max(dist(1, :));
         min_1=min(dist(1, :));
        sigmas = var(dist(1, :));
        weights(i) = exp(-(diss-min_1)/(max_1-min_1)*sigmas);
        end
    else
        % 属于第二簇类
        dis = dist(2, i);
        max_2=max(dist(2, :));
         min_2=min(dist(2, :));
        sigma = var(dist(2, :));
          weights(i) = exp(-(dis-min_2)/(max_2-min_2)*sigma);
    end
end
%         cluster_1_membership = U(1, :);
% cluster_2_membership = U(2, :);
% 
% % 初始化特征通道权重为1
% feature_weights = ones(1, size(x,3));
% 
% % 将属于第二簇类的特征通道权重设为0.99
% feature_weights(cluster_2_membership > cluster_1_membership) = 0.91;
% 
% % 将属于第一簇类的特征通道隶属度大于0.7的通道权重设为1
% feature_weights((cluster_1_membership >= cluster_2_membership) & (cluster_1_membership > 0.7)) = 1;
% 
% % 将属于第一簇类的特征通道隶属度小于0.7的通道权重设为0.99
% feature_weights((cluster_1_membership >= cluster_2_membership) & (cluster_1_membership <= 0.7)) = 0.99;
%   x=reshape(feature_weights, 1, 1, []).*x;

    x=reshape(weights, 1, 1, []).*x;
      ct_m = mexResize(context_m,[size(x,1) size(x,2)],'auto');
        xc = x .* ct_m;
        xf=fft2(bsxfun(@times, x, cos_window));
        xcf_c=fft2(bsxfun(@times, xc, cos_window));
        xcf_p = zeros(size(xcf_c));
         xcf_pp = zeros(size(xcf_c));
        model_xf = xf;
        xcf = xcf_c - xcf_p;
         [g_f] = run_training(model_xf, xcf, use_sz, params,yf, smallsz);
           xcf_p = xcf_c; 
    elseif frame==2
        % use detection features
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
        xcf_c = shift_sample(xcf_c, shift_samp_pos, kx', ky');
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
        xcf = xcf_c - xcf_p;
        [g_f] = run_training(model_xf, xcf, use_sz, params,yf, smallsz);
    xcf_pp=xcf_p;
    xcf_p = xcf_c;
        else
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
        xcf_c = shift_sample(xcf_c, shift_samp_pos, kx', ky');
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);      
        
    end
    
    % context residual
   xcf = xcf_c - 2*xcf_p+xcf_pp;
    
    [g_f] = run_training(model_xf, xcf, use_sz, params,yf, smallsz);
    
      xcf_pp=xcf_p;
    xcf_p = xcf_c;
    
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
end
%   show speed
disp(['fps: ' num2str(results.fps)])
