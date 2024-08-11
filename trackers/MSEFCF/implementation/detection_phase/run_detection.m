% function [xtf,xtcf,pos,translation_vec,response,disp_row,disp_col] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame)
%     center=pos+[Vy Vx];
% %    center=pos;
%     pixel_template=get_pixels(im, center, round(sz*currentScaleFactor), sz);  
% %         figure(10);
% %         imshow(pixel_template);
% % %         
% %              figure(11);
% %              imshow(m);
%     m = context_mask(pixel_template,round(target_sz/currentScaleFactor));
%     xt = get_features(pixel_template,features,global_feat_params);
% %             num=size(xt,1);
% %                   num_factors = factor(num);  % 获取数字的质因数分解
% %     gcd_except_self = max(num_factors(num_factors ~= num))
% %         small_array_size=gcd_except_self;
% % xt = restore_features(xt, small_array_size);
%     inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
% % % 找到 OB 矩阵中不为 1 的元素的行和列索引
% % [rows, cols] = find(inverse_m  ~= 1);
% % 
% % % 计算行的范围
% % min_row = min(rows);
% % max_row = max(rows);
% % 
% % % 计算列的范围
% % min_col = min(cols);
% % max_col = max(cols);
% % % 假设 x、M、min_col、min_row 和 max_col 已经定义好
% % M=size(xt,1);
% % % 分割的第一个矩阵（0，M）行范围，（0，min_col）列范围部分
% % block1 = xt(1:M, 1:min_col, :);
% % 
% % % 分割的第二个矩阵（0，min_row）行范围，（min_col，max_col）列范围部分
% % block2 = xt(1:min_row, min_col+1:max_col, :);
% % 
% % % 分割的第三个矩阵（min_row，max_row）行范围，（min_col，max_col）列范围部分
% % block3 = xt(min_row+1:max_row, min_col+1:max_col, :);
% % 
% % % 分割的第四个矩阵（max_row，M）行范围，（min_col，max_col）列范围部分
% % block4 = xt(max_row+1:M, min_col+1:max_col, :);
% % 
% % % 分割的第五个矩阵（0，M）行范围，（max_col，M）列范围部分
% % block5 = xt(1:M, max_col+1:end, :);
% % 
% % weighted_block1 = applyWeighting(block1);
% % weighted_block2 = applyWeighting(block2);
% % weighted_block3= block3;
% % weighted_block4 = applyWeighting(block4);
% % weighted_block5 = applyWeighting(block5);
% %        
% % % 假设你已经定义了所有的分割块 weighted_block1 到 weighted_block5
% % 
% % % 拼接 weighted_block2、weighted_block3 和 weighted_block4
% % weighted_A = cat(1, weighted_block2, weighted_block3, weighted_block4);
% % 
% % % 计算拼接后的行数和列数
% % merged_rows = size(weighted_A, 1);
% % merged_cols = size(weighted_block1, 2) + size(weighted_A, 2) + size(weighted_block5, 2);
% % 
% % % 创建一个新的矩阵来保存最终的结果
% % new_x = zeros(merged_rows, merged_cols, size(weighted_block1, 3));
% % 
% % % 将拼接后的分割块放入最终的结果矩阵中
% % new_x(:, 1:size(weighted_block1, 2), :) = weighted_block1;
% % new_x(:, size(weighted_block1, 2)+1:size(weighted_block1, 2)+size(weighted_A, 2), :) = weighted_A;
% % new_x(:, size(weighted_block1, 2)+size(weighted_A, 2)+1:end, :) = weighted_block5;
% % xt=new_x;
% %     
% %     
% %     
%     
%     
%     
%     
%     
% %      figure(11);
% %      imshow(m);
%      
% %     xtc = xt .* inverse_m;
% xtc = xt;
%     xtf = fft2(bsxfun(@times,xt,cos_window));    
%     xtcf = fft2(bsxfun(@times,xtc,cos_window)); 
% %         savedir='H:\IROS\Ablation\features\';
% %         if frame==295
% %         xt_f=ifft2(xtf,'symmetric');
% %         Xt=sum(xt_f,3);
% %         colormap(jet);
% %         surf(Xt);
% %         shading interp;
% %         axis ij;
% %         axis off;
% %         view([34,50]);
% %         saveas(gcf,[savedir,num2str(frame),'.png']);
% %         end
% % %             savedir='H:\IROS\DR2Track\DR2_JOURNAL\Fig1\Featuremaps\';
% %             if frame==49
% %                 for i=1:42
% %             set(gcf,'visible','off'); 
% %             colormap(parula);
% %             Q=surf(xt(:,:,i));
% %             axis ij;
% %             axis off;
% %             view([0,90]);
% %             set(Q,'edgecolor','none');
% % %             shading interp
% %             saveas(gcf,[savedir,num2str(i),'.png']);
% %                 end
% %             end
%     responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
%     % if we undersampled features, we want to interpolate the
%     % response so it has the same size as the image patch
% 
%     responsef_padded = resizeDFT2(responsef, use_sz);
%     % response in the spatial domain
%     response = ifft2(responsef_padded, 'symmetric');
% %         figure(15),surf(fftshift(response));
% 
%     % find maximum peak
%     [disp_row, disp_col] = resp_newton(response, responsef_padded,newton_iterations, ky, kx, use_sz);
%     % calculate translation
%     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
%     %update position
%     pos = center + translation_vec;
% end
% 
% function [xtf,xtcf,pos,translation_vec,response,disp_row,disp_col] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame)
%     center=pos+[Vy Vx];
% %    center=pos;
%     pixel_template=get_pixels(im, center, round(sz*currentScaleFactor), sz);  
% %         figure(10);
% %         imshow(pixel_template);
% % %         
% %              figure(11);
% %              imshow(m);
%     m = context_mask(pixel_template,round(target_sz/currentScaleFactor));
%     xt = get_features(pixel_template,features,global_feat_params);
%     inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
%     [rows, cols] = find(inverse_m ~= 1);
% shifan = xt(:,:,42);
% % 假设 x 是你的三维特征数组，ct_m 是表示目标区域的二维矩阵
% % 假设 M 是数组的维度大小，D 是通道数目
% 
% % 计算行的范围
% min_row = min(rows);
% max_row = max(rows);
% 
% % 计算列的范围
% min_col = min(cols);
% max_col = max(cols);
%     global_mean = mean(xt(:), 'all');
%     global_std = std(xt(:), 0, 'all');
% % 遍历 ct_m 矩阵，提取目标区域并应用多尺度增强
% for d = 1:size(xt, 3)
%     target_feature = xt(min_row:max_row, min_col:max_col, d);
%     
%     % 计算整个通道的全局均值和标准差
% %     global_mean = mean(xt(:,:,d), 'all');
% %     global_std = std(xt(:,:,d), 0, 'all');
% if std(target_feature(:)) > 1e-6
%     global_enhanced_target = (target_feature - mean(target_feature(:))) * (global_std / std(target_feature(:))) + global_mean;
% else
%     global_enhanced_target = target_feature;
% end
%     % 定义多尺度参数
% %     scales = [1, 0.8, 0.6]; % 尺度因子
%     scales = [1]; % 如果只使用单一尺度
% 
%     % 多尺度增强
%     for s = 1:length(scales)
%         % 缩小图像尺寸
%         scaled_target = imresize(target_feature, scales(s));
% if std(scaled_target(:)) > 1e-6
%         % 在缩小的图像上应用增强
%         enhanced_target = (scaled_target - mean(scaled_target(:))) * (global_std / std(scaled_target(:))) + global_mean;
% else
%      enhanced_target=scaled_target;
% end
%         % 将增强后的小图像放大回原始尺寸
%         enhanced_target = imresize(enhanced_target, size(target_feature));
% 
%         % 在全局增强和多尺度增强之间加权融合
%         alpha = 0.5; % 加权因子
%         blended_enhanced = alpha * global_enhanced_target + (1 - alpha) * enhanced_target;
% 
%         % 将处理后的特征放回到原始数组中的相应位置
%         xt(min_row:max_row, min_col:max_col, d) = blended_enhanced;
%     end
% end
% 
% shifanfan = xt(:,:,42); % 获取增强后的特征
% %     
%     
%     
%     
%     
%     % 假设 x 是你的三维特征数组，ct_m 是表示目标区域的二维矩阵
% % 假设 M 是数组的维度大小，D 是通道数目
% % [rows, cols] = find(inverse_m  ~= 1);
% % 
% % % 计算行的范围
% % min_row = min(rows);
% % max_row = max(rows);
% % 
% % % 计算列的范围
% % min_col = min(cols);
% % max_col = max(cols);
% % % 创建存储结果的数组
% % enhanced_xt = xt;
% % % 遍历 ct_m 矩阵，提取目标区域并应用 adapthisteq
% % 
% %             % 提取目标区域特征子数组
% %             target_feature = xt(min_row:max_row, min_col:max_col, :);
% %             
% %             % 对每个通道进行处理
% %             for d = 1:size(xt,3)
% %                 target_image = target_feature(:, :, d);
% %                 
% %                 num_levels=64;
% %                 % 应用 adapthisteq
% %                 enhanced_target = histeq(target_image,num_levels);
% %                 
% %                 % 将处理后的特征放回到原始数组中的相应位置
% %                 enhanced_xt(min_row:max_row, min_col:max_col, d) = enhanced_target;
% %             end
% % after= enhanced_xt(:,:,42);
% % xt=enhanced_xt;
% %      figure(11);
% %      imshow(m);
%      
%     xtc = xt .* inverse_m;
% % xtc = xt;
%     xtf = fft2(bsxfun(@times,xt,cos_window));    
%     xtcf = fft2(bsxfun(@times,xtc,cos_window)); 
% %         savedir='H:\IROS\Ablation\features\';
% %         if frame==295
% %         xt_f=ifft2(xtf,'symmetric');
% %         Xt=sum(xt_f,3);
% %         colormap(jet);
% %         surf(Xt);
% %         shading interp;
% %         axis ij;
% %         axis off;
% %         view([34,50]);
% %         saveas(gcf,[savedir,num2str(frame),'.png']);
% %         end
% % %             savedir='H:\IROS\DR2Track\DR2_JOURNAL\Fig1\Featuremaps\';
% %             if frame==49
% %                 for i=1:42
% %             set(gcf,'visible','off'); 
% %             colormap(parula);
% %             Q=surf(xt(:,:,i));
% %             axis ij;
% %             axis off;
% %             view([0,90]);
% %             set(Q,'edgecolor','none');
% % %             shading interp
% %             saveas(gcf,[savedir,num2str(i),'.png']);
% %                 end
% %             end
%     responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
%     % if we undersampled features, we want to interpolate the
%     % response so it has the same size as the image patch
% 
%     responsef_padded = resizeDFT2(responsef, use_sz);
%     % response in the spatial domain
%     response = ifft2(responsef_padded, 'symmetric');
% %         figure(15),surf(fftshift(response));
% 
%     % find maximum peak
%     [disp_row, disp_col] = resp_newton(response, responsef_padded,newton_iterations, ky, kx, use_sz);
%     % calculate translation
%     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
%     %update position
%     pos = center + translation_vec;
% end
% 
% function [xtf,xtcf,pos,translation_vec,response,disp_row,disp_col] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame)
%     center=pos+[Vy Vx];
% %    center=pos;
%     pixel_template=get_pixels(im, center, round(sz*currentScaleFactor), sz);  
% %         figure(10);
% %         imshow(pixel_template);
% % %         
% %              figure(11);
% %              imshow(m);
%     m = context_mask(pixel_template,round(target_sz/currentScaleFactor));
%     xt = get_features(pixel_template,features,global_feat_params);
%     inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
%     [rows, cols] = find(inverse_m ~= 1);
% 
% % 计算行的范围
% min_row = min(rows);
% max_row = max(rows);
% 
% % 计算列的范围
% min_col = min(cols);
% max_col = max(cols);
% 
% % 在全局增强之前，计算整个特征数组的全局均值和标准差
% global_mean = mean(xt(:));
% global_std = std(xt(:));
% 
% % 遍历 ct_m 矩阵，提取目标区域并应用多尺度增强
% for d = 1:size(xt, 3)
%     target_feature = xt(min_row:max_row, min_col:max_col, d);
% 
%     % 全局增强
%     global_enhanced_target = (target_feature - mean(target_feature(:))) * (global_std / std(target_feature(:))) + global_mean;
% % 定义多尺度参数
% scales = [1, 0.8, 0.6]; % 尺度因子
% 
%     % 多尺度增强
%     for s = 1:length(scales)
%         % 缩小图像尺寸
%         scaled_target = imresize(target_feature, scales(s));
% 
%         % 在缩小的图像上应用增强
%         enhanced_target = (scaled_target - mean(scaled_target(:))) * (global_std / std(scaled_target(:))) + global_mean;
% 
%         % 将增强后的小图像放大回原始尺寸
%         enhanced_target = imresize(enhanced_target, size(target_feature));
% 
%         % 在全局增强和多尺度增强之间加权融合
%         alpha = 0.5; % 加权因子
%         blended_enhanced = alpha * global_enhanced_target + (1 - alpha) * enhanced_target;
% 
%         % 将处理后的特征放回到原始数组中的相应位置
%         xt(min_row:max_row, min_col:max_col, d) = blended_enhanced;
%     end
% end
%     
%     
%     
%     
%     
%     
%     
%     % 假设 x 是你的三维特征数组，ct_m 是表示目标区域的二维矩阵
% % 假设 M 是数组的维度大小，D 是通道数目
% % [rows, cols] = find(inverse_m  ~= 1);
% % 
% % % 计算行的范围
% % min_row = min(rows);
% % max_row = max(rows);
% % 
% % % 计算列的范围
% % min_col = min(cols);
% % max_col = max(cols);
% % % 创建存储结果的数组
% % enhanced_xt = xt;
% % % 遍历 ct_m 矩阵，提取目标区域并应用 adapthisteq
% % 
% %             % 提取目标区域特征子数组
% %             target_feature = xt(min_row:max_row, min_col:max_col, :);
% %             
% %             % 对每个通道进行处理
% %             for d = 1:size(xt,3)
% %                 target_image = target_feature(:, :, d);
% %                 
% %                 num_levels=64;
% %                 % 应用 adapthisteq
% %                 enhanced_target = histeq(target_image,num_levels);
% %                 
% %                 % 将处理后的特征放回到原始数组中的相应位置
% %                 enhanced_xt(min_row:max_row, min_col:max_col, d) = enhanced_target;
% %             end
% % after= enhanced_xt(:,:,42);
% % xt=enhanced_xt;
% %      figure(11);
% %      imshow(m);
%      
%     xtc = xt .* inverse_m;
% % xtc = xt;
%     xtf = fft2(bsxfun(@times,xt,cos_window));    
%     xtcf = fft2(bsxfun(@times,xtc,cos_window)); 
% %         savedir='H:\IROS\Ablation\features\';
% %         if frame==295
% %         xt_f=ifft2(xtf,'symmetric');
% %         Xt=sum(xt_f,3);
% %         colormap(jet);
% %         surf(Xt);
% %         shading interp;
% %         axis ij;
% %         axis off;
% %         view([34,50]);
% %         saveas(gcf,[savedir,num2str(frame),'.png']);
% %         end
% % %             savedir='H:\IROS\DR2Track\DR2_JOURNAL\Fig1\Featuremaps\';
% %             if frame==49
% %                 for i=1:42
% %             set(gcf,'visible','off'); 
% %             colormap(parula);
% %             Q=surf(xt(:,:,i));
% %             axis ij;
% %             axis off;
% %             view([0,90]);
% %             set(Q,'edgecolor','none');
% % %             shading interp
% %             saveas(gcf,[savedir,num2str(i),'.png']);
% %                 end
% %             end
%     responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
%     % if we undersampled features, we want to interpolate the
%     % response so it has the same size as the image patch
% 
%     responsef_padded = resizeDFT2(responsef, use_sz);
%     % response in the spatial domain
%     response = ifft2(responsef_padded, 'symmetric');
% %         figure(15),surf(fftshift(response));
% 
%     % find maximum peak
%     [disp_row, disp_col] = resp_newton(response, responsef_padded,newton_iterations, ky, kx, use_sz);
%     % calculate translation
%     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
%     %update position
%     pos = center + translation_vec;
% end
% 
function [xtf,xtcf,pos,translation_vec,response,disp_row,disp_col] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame)
    center=pos+[Vy Vx];
%    center=pos;
    pixel_template=get_pixels(im, center, round(sz*currentScaleFactor), sz);  
%         figure(10);
%         imshow(pixel_template);
% %         
%              figure(11);
%              imshow(m);
    m = context_mask(pixel_template,round(target_sz/currentScaleFactor));
    xt = get_features(pixel_template,features,global_feat_params);
    shifan=xt(:,:,42);
    inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
% 假设 x 是你的三维特征数组，ct_m 是表示目标区域的二维矩阵
% 假设 M 是数组的维度大小，D 是通道数目
[rows, cols] = find( inverse_m ~= 1);

% 计算行的范围
min_row = min(rows);
max_row = max(rows);

% 计算列的范围
min_col = min(cols);
max_col = max(cols);
% 全局增强参数
global_enhance_factor = 1.48;

% 多尺度增强参数
scales = [0.5, 1, 1.5]; % 可根据需要调整尺度

% 对每个通道进行增强
enhanced_xt = xt;
for d = 1:size(xt,3)
    % 提取目标区域特征
    target_feature = enhanced_xt(:,:,d);

    % 计算目标区域的均值和标准差
    target_mean = mean(target_feature(:));
    target_std = std(target_feature(:));

    % 全局增强
    enhanced_xt(:,:,d) = (enhanced_xt(:,:,d) - target_mean) * global_enhance_factor + target_mean;
    global_xt=enhanced_xt;
    % 多尺度增强
    for scale_idx = 1:length(scales)
        scaled_target = imresize(target_feature, scales(scale_idx));

        % 计算当前尺度下的均值和标准差
        scaled_mean = mean(scaled_target(:));
        scaled_std = std(scaled_target(:));

        % 将当前尺度的特征映射应用到相应的区域
        if scaled_std > 1e-6
        scaled_enhanced_feature = (enhanced_xt(min_row:max_row, min_col:max_col, d) - scaled_mean) * (target_std / scaled_std) + scaled_mean;
        % 更新增强后的特征
        else
         scaled_enhanced_feature=enhanced_xt(min_row:max_row, min_col:max_col, d);
        end
        enhanced_xt(min_row:max_row, min_col:max_col, d) = scaled_enhanced_feature;
        alpha=0.3;
         enhanced_xt=alpha*global_xt+(1-alpha)* enhanced_xt;
        
    end
end

% 最终增强后的特征数组
enhanced_features = enhanced_xt;
xt=enhanced_features;
shifanfan = xt(:,:,42); % 获取增强后的特征

    
    
    
    % 假设 x 是你的三维特征数组，ct_m 是表示目标区域的二维矩阵
% 假设 M 是数组的维度大小，D 是通道数目
% [rows, cols] = find(inverse_m  ~= 1);
% 
% % 计算行的范围
% min_row = min(rows);
% max_row = max(rows);
% 
% % 计算列的范围
% min_col = min(cols);
% max_col = max(cols);
% % 创建存储结果的数组
% enhanced_xt = xt;
% % 遍历 ct_m 矩阵，提取目标区域并应用 adapthisteq
% 
%             % 提取目标区域特征子数组
%             target_feature = xt(min_row:max_row, min_col:max_col, :);
%             
%             % 对每个通道进行处理
%             for d = 1:size(xt,3)
%                 target_image = target_feature(:, :, d);
%                 
%                 num_levels=64;
%                 % 应用 adapthisteq
%                 enhanced_target = histeq(target_image,num_levels);
%                 
%                 % 将处理后的特征放回到原始数组中的相应位置
%                 enhanced_xt(min_row:max_row, min_col:max_col, d) = enhanced_target;
%             end
% after= enhanced_xt(:,:,42);
% xt=enhanced_xt;
%      figure(11);
%      imshow(m);
     
%     xtc = xt .* inverse_m;
xtc = xt;
    xtf = fft2(bsxfun(@times,xt,cos_window));    
    xtcf = fft2(bsxfun(@times,xtc,cos_window)); 
%         savedir='H:\IROS\Ablation\features\';
%         if frame==295
%         xt_f=ifft2(xtf,'symmetric');
%         Xt=sum(xt_f,3);
%         colormap(jet);
%         surf(Xt);
%         shading interp;
%         axis ij;
%         axis off;
%         view([34,50]);
%         saveas(gcf,[savedir,num2str(frame),'.png']);
%         end
% %             savedir='H:\IROS\DR2Track\DR2_JOURNAL\Fig1\Featuremaps\';
%             if frame==49
%                 for i=1:42
%             set(gcf,'visible','off'); 
%             colormap(parula);
%             Q=surf(xt(:,:,i));
%             axis ij;
%             axis off;
%             view([0,90]);
%             set(Q,'edgecolor','none');
% %             shading interp
%             saveas(gcf,[savedir,num2str(i),'.png']);
%                 end
%             end
    responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
    % if we undersampled features, we want to interpolate the
    % response so it has the same size as the image patch

    responsef_padded = resizeDFT2(responsef, use_sz);
    % response in the spatial domain
    response = ifft2(responsef_padded, 'symmetric');
%         figure(15),surf(fftshift(response));

    % find maximum peak
    [disp_row, disp_col] = resp_newton(response, responsef_padded,newton_iterations, ky, kx, use_sz);
    % calculate translation
    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
    %update position
    pos = center + translation_vec;
end



