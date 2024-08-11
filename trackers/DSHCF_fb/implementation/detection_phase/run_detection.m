function [xtf,xtcf,pos,translation_vec,response,disp_row,disp_col] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame)
    center=pos+[Vy Vx];
%    center=pos;
    pixel_template=get_pixels(im, center, round(sz*currentScaleFactor), sz); 

%         figure(10);
%         imshow(pixel_template);
% %         
%              figure(11);
%              imshow(m);
%     % 应用双边滤波进行去噪
%     ksize = 1;  % 定义滤波核大小
%     sigmac = 2;  % 定义空间域的标准差
%     sigmas = 25;  % 定义值域的标准差
%     
    % 调用双边滤波函数
%     denoised_template = bifilter(ksize, sigmac, sigmas, pixel_template);
%     denoised_template = imgaussfilt(pixel_template, 1);
%     denoised_template = fastBilateralFilter(pixel_template, sigmac, sigmas, 4);

%     noisyRGB = imnoise(pixel_template,'gaussian',0,0.0015);
%     noisyLAB = rgb2lab(noisyRGB);
%     patchSq = noisyLAB.^2;
%     edist = sqrt(sum(patchSq,3));
%     patchSigma = sqrt(var(edist(:)));
%     DoS = 1.5*patchSigma;
%     denoisedLAB = imnlmfilt(noisyLAB, 'DegreeOfSmoothing', DoS, 'SearchWindowSize', 7, 'ComparisonWindowSize', 3);  % 默认通常为21,7
%     denoised_img = lab2rgb(denoisedLAB,'Out','uint8');

%     patchSize = 5;        % 比较窗口的大小
%     windowSize = 11;      % 搜索窗口的大小
%     filterStrength = 0.2; % 过滤强度，决定了去噪的程度
%     denoised_img = imnlmfilt(pixel_template, 'DegreeOfSmoothing', filterStrength, ...
%                             'ComparisonWindowSize', patchSize, ...
%                             'SearchWindowSize', windowSize);
%     denoised_template = histeq(pixel_template);

%     % 对每个颜色通道分别应用中值滤波
%     for k = 1:3  % 遍历每个颜色通道
%         denoised_img(:,:,k) = medfilt2(pixel_template(:,:,k), [3 3]);
%     end
% 
%     lambda = 1;  % 正则化参数
%     rho = 0.01;     % ADMM参数
%     num_iterations = 7;  % 迭代次数
    
%     denoised_img = admm_denoise_color(pixel_template, lambda, rho, num_iterations);

%     denoised_img = tv_color(pixel_template, 7);
%     figure;
%     subplot(1,2,1);
%     imshow(pixel_template);
%     title('Original Image');
%     
%     subplot(1,2,2);
%     imshow(denoised_img);
%     title('Denoised Image using ADMM');

%     图像对比度增强
%     shadow_lab = rgb2lab(double(pixel_template));
%     max_luminosity = 100;
%     L = shadow_lab(:,:,1)/max_luminosity;
% %     shadow_imadjust = shadow_lab;
% %     shadow_imadjust(:,:,1) = imadjust(L)*max_luminosity;
% %     shadow_imadjust = lab2rgb(shadow_imadjust);
% %     
%     shadow_histeq = shadow_lab;
%     shadow_histeq(:,:,1) = histeq(L)*max_luminosity;
%     shadow_histeq = lab2rgb(shadow_histeq);
    
%     shadow_adapthisteq = shadow_lab;
%     shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
%     shadow_adapthisteq = lab2rgb(shadow_adapthisteq);
%     denoised_img = lab2rgb(shadow_histeq,'Out','uint8');

% % Wiener滤波
%     PSF = fspecial('motion',21,0.1);
%     wnr1 = deconvwnr(double(pixel_template),PSF);
%     denoised_img = im2uint8(wnr1);  % 直接转换为 uint8 类型

    % 对每个颜色通道分别应用wiener2滤波
%     for k = 1:3  % 遍历每个颜色通道
%         denoised_img(:,:,k) = imsharpen(pixel_template(:,:,k));
%     end
    denoised_img = imsharpen(pixel_template);
    % 以下代码继续使用去噪后的图像
    m = context_mask(denoised_img, round(target_sz/currentScaleFactor));
    xt = get_features(denoised_img, features, global_feat_params);

%     m = context_mask(pixel_template,round(target_sz/currentScaleFactor));
%     xt = get_features(pixel_template,features,global_feat_params);
    inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
    
%      figure(11);
%      imshow(m);
     
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

