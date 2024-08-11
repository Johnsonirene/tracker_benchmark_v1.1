function out = bifilter(ksize, sigmac, sigmas, in)
    % 参数分别指定kernel size、空间域的标准差、值域的标准差和原始图像
    
    [h, w, d] = size(in);
    % 获取输入图像的尺寸和通道数
    
    % 创建一个以kernel中心为原点的坐标网格
    [X, Y] = meshgrid(-floor(ksize/2):floor(ksize/2), -floor(ksize/2):floor(ksize/2));
    dist = X.^2 + Y.^2;
    Gc = exp(-dist/(2*sigmac^2));
    % 计算空间权重，这是基于距离的高斯函数
    
    out = zeros(size(in), 'like', in); % 初始化输出图像

    padSize = floor(ksize/2);
    padim = padarray(in, [padSize, padSize], 'replicate', 'both');
    % 对图像进行边界扩展，使用'replicate'避免引入0值导致边缘效应
%     disp(['padim size: ', num2str(size(padim))]);
    for i = 1:h
        for j = 1:w
            % 提取当前像素周围的区域
            localRegion = double(padim(i:i+ksize-1, j:j+ksize-1, :));
            
            % 修正边缘位置的处理
            centerPixel = localRegion(padSize+1, padSize+1, :);
            
            for k = 1:d
                tempDiff = localRegion(:,:,k) - centerPixel(:,:,k);
                Gs = exp(-tempDiff.^2 / (2 * sigmas^2));
                Wp = sum(sum(Gc .* Gs));
                out(i,j,k) = sum(sum(Gc .* Gs .* localRegion(:,:,k))) / Wp;
            end
        end
    end
    
    out = uint8(out); % 转换为8位无符号整数
end
