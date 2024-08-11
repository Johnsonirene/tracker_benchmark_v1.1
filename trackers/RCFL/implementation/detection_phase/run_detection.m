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
    inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
ORGIN=xt(:,:,1);
%      figure(11);
%      imshow(m);
       sigma = 0.3; % 高斯滤波器的标准差
        xxt = imgaussfilt3(xt, sigma);
         later=xt(:,:,1);
         for i=size(xt,3)
% 计算特征中心的位置（假设特征中心位于 (25, 25)）

featureChannel=xt(:,:,i);
noisyImage=xxt(:,:,i);

% 结合特征和分水岭算法
% combinedImage = imfuse(noisyImage, featureChannel, 'blend', 'Scaling', 'none'); % 使用imfuse函数进行图像融合，并禁用灰度缩放
% alpha = 0.5;  % 自定义权重
% combinedImage = imfuse(noisyImage, featureChannel, 'blend', 'Scaling', 'none', 'BlendAlpha', alpha);
% combinedImage = im2single(combinedImage); % 将图像类型转换为single
alpha = 0.5;  % 自定义权重
combinedImage = alpha * noisyImage + (1 - alpha) * featureChannel;

 xt(:,:,i)=combinedImage;
 

% 使用分水岭算法进行区域分割，并引入距离惩罚项
        end
% 设置高斯滤波的参数，sigma为标准偏差，hsize表示滤波器大小
% sigma = 0.3;
% hsize = [3,3];

% 对每个通道进行高斯滤波
% for i = 1:size(xt,3) % D为通道数
%     xt(:,:,i) = imgaussfilt(xt(:,:,i), sigma, 'FilterSize', hsize);
% end
    xtc = xt .* inverse_m;
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

