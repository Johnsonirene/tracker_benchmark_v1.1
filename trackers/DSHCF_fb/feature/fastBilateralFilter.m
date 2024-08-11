function out = fastBilateralFilter(img, sigmaColor, sigmaSpace, downsampleFactor)

    ksize = 1;
    % 下采样图像
    smallImg = imresize(img, 1/downsampleFactor, 'bicubic');
    
    % 应用传统的双边滤波
    smallFilteredImg = bifilter(ksize, sigmaColor, sigmaSpace, smallImg);
    
    % 上采样到原始尺寸
    out = imresize(smallFilteredImg, size(img(:,:,1)), 'bicubic');
end
