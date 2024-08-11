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
         ct_m = mexResize(context_m,[size(x,1) size(x,2)],'auto');
%          total_elements = numel(ct_m);
% count_ones = sum(ct_m(:) == 1);
% count_not_ones = total_elements - count_ones;
% 
% p1 = count_ones / total_elements;
% p2 = count_not_ones / total_elements;
% 
% ct_m_processed = ct_m;
% ct_m_processed(ct_m == 1) = p1;
% ct_m_processed(ct_m ~= 1) = p2;
         shifan=ones(size(x,1),size(x,2));
% 找到 OB 矩阵中不为 1 的元素的行和列索引
[rows, cols] = find(ct_m  ~= 1);

% 计算行的范围
min_row = min(rows);
max_row = max(rows);

% 计算列的范围
min_col = min(cols);
max_col = max(cols);
% 假设 x、M、min_col、min_row 和 max_col 已经定义好
M=size(x,1);
% 分割的第一个矩阵（0，M）行范围，（0，min_col）列范围部分
block1 = x(1:M, 1:min_col, :);

% 分割的第二个矩阵（0，min_row）行范围，（min_col，max_col）列范围部分
block2 = x(1:min_row, min_col+1:max_col, :);

% 分割的第三个矩阵（min_row，max_row）行范围，（min_col，max_col）列范围部分
block3 = x(min_row+1:max_row, min_col+1:max_col, :);

% 分割的第四个矩阵（max_row，M）行范围，（min_col，max_col）列范围部分
block4 = x(max_row+1:M, min_col+1:max_col, :);

% 分割的第五个矩阵（0，M）行范围，（max_col，M）列范围部分
block5 = x(1:M, max_col+1:end, :);

weighted_block1 = applyWeighting(block1);
weighted_block2 = applyWeighting(block2);
weighted_block3= block3;
weighted_block4 = applyWeighting(block4);
weighted_block5 = applyWeighting(block5);
       
% 假设你已经定义了所有的分割块 weighted_block1 到 weighted_block5

% 拼接 weighted_block2、weighted_block3 和 weighted_block4
weighted_A = cat(1, weighted_block2, weighted_block3, weighted_block4);

% 计算拼接后的行数和列数
merged_rows = size(weighted_A, 1);
merged_cols = size(weighted_block1, 2) + size(weighted_A, 2) + size(weighted_block5, 2);

% 创建一个新的矩阵来保存最终的结果
new_x = zeros(merged_rows, merged_cols, size(weighted_block1, 3));

% 将拼接后的分割块放入最终的结果矩阵中
new_x(:, 1:size(weighted_block1, 2), :) = weighted_block1;
new_x(:, size(weighted_block1, 2)+1:size(weighted_block1, 2)+size(weighted_A, 2), :) = weighted_A;
new_x(:, size(weighted_block1, 2)+size(weighted_A, 2)+1:end, :) = weighted_block5;
x=new_x;
% 结果保存在 new_x 中

% 

%         a=size(x,1);
%      num = zuidagongyueshu(a);
%         small_array_size=num;
%        x= restore_features(x, small_array_size);
%         ct_m = mexResize(context_m,[size(x,1) size(x,2)],'auto');
%         xc = x .* ct_m;
           xc = x;
        xf=fft2(bsxfun(@times, x, cos_window));
        xcf_c=fft2(bsxfun(@times, xc, cos_window));
        xcf_p = zeros(size(xcf_c));
         xcf_pp = zeros(size(xcf_c));
         g_f_p= zeros(size(xcf_c));
         g_f_pp= zeros(size(xcf_c));
        gcf= g_f_p;
          xcf = xcf_c - xcf_p;
        model_xf = xf;
          [g_f] = run_training(model_xf,gcf,xcf, use_sz, params,yf, smallsz);

        xcf_p = xcf_c;
        g_f_p= g_f;
   elseif frame==2
        % use detection features
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
        xcf_c = shift_sample(xcf_c, shift_samp_pos, kx', ky');
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
                  xcf = xcf_c - xcf_p;
                  gcf=g_f_p;
        [g_f] = run_training(model_xf,gcf,xcf, use_sz, params,yf, smallsz);
   xcf_pp=xcf_p;
    xcf_p = xcf_c;
   g_f_pp =g_f_p;
     g_f_p =g_f;
    else
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
        xcf_c = shift_sample(xcf_c, shift_samp_pos, kx', ky');
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);      
        
    end
    
    % context residual
   xcf = xcf_c - 2*xcf_p+xcf_pp;
    gcf=2*g_f_p-g_f_pp;
    [g_f] = run_training(model_xf,gcf, xcf, use_sz, params,yf, smallsz);
      xcf_pp=xcf_p;
    xcf_p = xcf_c;
 g_f_pp =g_f_p;
     g_f_p =g_f;
    
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
