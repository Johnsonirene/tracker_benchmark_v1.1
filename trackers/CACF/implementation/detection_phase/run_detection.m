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
     XX = reshape(xt, [], size(xt, 3));  
        XX=XX';
        data=XX;
        cluster_n=2;
      [cen,dist,U, obj_fcn] = FCMClust(data, cluster_n);
   for i = 1:size(xt,3)
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
% feature_weights = ones(1, size(xt,3));
% 
% % 将属于第二簇类的特征通道权重设为0.99
% feature_weights(cluster_2_membership > cluster_1_membership) = 0.92;
% 
% % 将属于第一簇类的特征通道隶属度大于0.7的通道权重设为1
% feature_weights((cluster_1_membership >= cluster_2_membership) & (cluster_1_membership > 0.7)) = 1;
% 
% % 将属于第一簇类的特征通道隶属度小于0.7的通道权重设为0.99
% feature_weights((cluster_1_membership >= cluster_2_membership) & (cluster_1_membership <= 0.7)) = 0.99;
%   xt=reshape(feature_weights, 1, 1, []).*xt;
%      figure(11);
%      imshow(m);
      xt=reshape(weights, 1, 1, []).*xt;
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

