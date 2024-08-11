% function weighted_block = applyWeighting(block)
%     n = size(block, 1);
%     m = size(block, 2);
%     c = size(block, 3);
% 
%     reshaped_data = reshape(block, [], c)';
%     
%     % 计算中心点
%     center = mean(reshaped_data, 1);
% 
%     % 计算每个矩阵到中心点的距离
%     distances = sqrt(sum((reshaped_data - center).^2, 2));
% 
%     % 计算距离的标准差
%     std_distance = std(distances);
% 
% 
%     % 进行百分之99水平的Z检验，设置权重
%     weights = ones(1, c);
%     weights = exp(-(distances / max(distances) / std_distance));
% weights= 0.1 * weights + 0.9;
%     % 应用权重
%     weighted_block = reshape(weights, 1, 1, c) .* block;
% end
function weighted_block = applyWeighting(block)
    n = size(block, 1);
    m = size(block, 2);
    c = size(block, 3);
    reshaped_data = reshape(block, [], size(block, 3));

    % Perform K-Means clustering with k = 1
    k = 1;
    [idx, centers] = kmeans(reshaped_data, k);

    % Find the index of the center channel
    [~, center_channel] = max(centers);


    % Return the center channel
    center_channel = reshape(block(:, :, center_channel), size(block, 1), size(block, 2));
 reshaped_center = reshape(center_channel, [], 1);
channel_distances = sqrt(sum((reshaped_data - reshaped_center).^2, 1));

    % Calculate total distance's standard deviation
    total_std_distance = std(channel_distances);
%     reshaped_data = reshape(block, [], c)';
% 
%     % 计算中心点
%     center = mean(reshaped_data, 1);
% 
%     % 计算每个矩阵到中心点的距离
%     distances = sqrt(sum((reshaped_data - reshaped_center).^2, 2));
% 
%     % 计算距离的标准差
%     std_distance = std(distances);
% 
    % 计算平均值的两倍标准差
    mean_plus_2std =  2.56 * total_std_distance;
% 
    % 初始化权重数组
    weights = ones(1, c);
weightrate=0.1;
    % 根据条件设置权重
    weights(channel_distances > mean_plus_2std) = exp(-(channel_distances(channel_distances > mean_plus_2std) / max(channel_distances) / total_std_distance));
    weights = weightrate * weights + (1-weightrate)*1;
% 
%     % 应用权重
    weighted_block = reshape(weights, 1, 1, c) .* block;
end