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
%             num=size(xt,1);
%                   num_factors = factor(num);  % 获取数字的质因数分解
%     gcd_except_self = max(num_factors(num_factors ~= num))
%         small_array_size=gcd_except_self;
% xt = restore_features(xt, small_array_size);
    inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
% % 找到 OB 矩阵中不为 1 的元素的行和列索引
% [rows, cols] = find(inverse_m  ~= 1);
% 
% % 计算行的范围
% min_row = min(rows);
% max_row = max(rows);
% 
% % 计算列的范围
% min_col = min(cols);
% max_col = max(cols);
% % 假设 x、M、min_col、min_row 和 max_col 已经定义好
% M=size(xt,1);
% % 分割的第一个矩阵（0，M）行范围，（0，min_col）列范围部分
% block1 = xt(1:M, 1:min_col, :);
% 
% % 分割的第二个矩阵（0，min_row）行范围，（min_col，max_col）列范围部分
% block2 = xt(1:min_row, min_col+1:max_col, :);
% 
% % 分割的第三个矩阵（min_row，max_row）行范围，（min_col，max_col）列范围部分
% block3 = xt(min_row+1:max_row, min_col+1:max_col, :);
% 
% % 分割的第四个矩阵（max_row，M）行范围，（min_col，max_col）列范围部分
% block4 = xt(max_row+1:M, min_col+1:max_col, :);
% 
% % 分割的第五个矩阵（0，M）行范围，（max_col，M）列范围部分
% block5 = xt(1:M, max_col+1:end, :);
% 
% weighted_block1 = applyWeighting(block1);
% weighted_block2 = applyWeighting(block2);
% weighted_block3= block3;
% weighted_block4 = applyWeighting(block4);
% weighted_block5 = applyWeighting(block5);
%        
% % 假设你已经定义了所有的分割块 weighted_block1 到 weighted_block5
% 
% % 拼接 weighted_block2、weighted_block3 和 weighted_block4
% weighted_A = cat(1, weighted_block2, weighted_block3, weighted_block4);
% 
% % 计算拼接后的行数和列数
% merged_rows = size(weighted_A, 1);
% merged_cols = size(weighted_block1, 2) + size(weighted_A, 2) + size(weighted_block5, 2);
% 
% % 创建一个新的矩阵来保存最终的结果
% new_x = zeros(merged_rows, merged_cols, size(weighted_block1, 3));
% 
% % 将拼接后的分割块放入最终的结果矩阵中
% new_x(:, 1:size(weighted_block1, 2), :) = weighted_block1;
% new_x(:, size(weighted_block1, 2)+1:size(weighted_block1, 2)+size(weighted_A, 2), :) = weighted_A;
% new_x(:, size(weighted_block1, 2)+size(weighted_A, 2)+1:end, :) = weighted_block5;
% xt=new_x;
%     
%     
%     
    
    
    
    
    
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

