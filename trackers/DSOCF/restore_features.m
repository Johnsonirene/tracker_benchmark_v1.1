function x = restore_features(x, small_array_size)
% 假设 x 是您提取的 M * M * D 维目标特征数组
M = size(x, 1);
D = size(x, 3);
% 计算小数组的数量
num_small_arrays = (M / small_array_size) ^ 2;

% 初始化多维数组来存储分割后的小数组
small_arrays = zeros(small_array_size, small_array_size, D, num_small_arrays);

% 分割目标特征
small_array_idx = 1;
for i = 1 : M/small_array_size
    for j = 1 : M/small_array_size
        small_array = x((i-1)*small_array_size+1 : i*small_array_size, ...
                        (j-1)*small_array_size+1 : j*small_array_size, :);
        small_arrays(:, :, :, small_array_idx) = small_array;
        
        % 添加条件判断，确保不会多出一个小数组
        if small_array_idx < num_small_arrays
            small_array_idx = small_array_idx + 1;
        else
            break; % 如果已经到达最后一个小数组，退出循环
        end
    end
    
    % 添加条件判断，确保不会多出一个小数组
    if small_array_idx == num_small_arrays
        break; % 如果已经到达最后一个小数组，退出外层循环
    end
end

% 假设 small_arrays 是你的 10*10*D 维度的数组，表示 D 个 10*10 的矩阵
% 假设 total_small_arrays 是你的方块的总数
total_small_arrays = num_small_arrays;
D = size(x, 3); % 获取特征维度

% 定义alpha数组用于存储权重
alpha = zeros(small_array_size, small_array_size, D, total_small_arrays);

for small_array_idx = 1:total_small_arrays
    % 获取第 small_array_idx 个方块的数据
    selected_array = small_arrays(:, :, :, small_array_idx);
        XX = reshape(selected_array, [], size(selected_array, 3));  
        XX=XX';
        data=XX;
        cluster_n=2;
        [cen, dist,U, obj_fcn] = FCMClust(data, cluster_n);
        % 初始化权重向量

% 对于每个特征通道，确定其属于哪个簇类
for i = 1:size(selected_array,3)
    if U(1, i) > U(2, i)
        % 属于第一簇类
        if U(1, i) > 0.7
            weights(i) = 1;
        else
         diss = dist(1, i);
         max_1=max(dist(1, :));
         min_1=min(dist(1, :));
        sigmas = var(dist(1, :));
%         weights(i) = exp(-(diss-min_1)/(max_1-min_1)/sigmas);
weights(i)=0.99;
        end
    else
        % 属于第二簇类
        dis = dist(2, i);
        max_2=max(dist(2, :));
         min_2=min(dist(2, :));
        sigma = var(dist(2, :));
%           weights(i) = exp(-(dis-min_2)/(max_2-min_2)/sigma)
weights(i)=0.97;
    end
end
%         % 将 D 个矩阵 reshape 成 D 行，每行是一个 10*10 的矩阵展开为一维向量
%     reshaped_data = reshape(selected_array, [], D)';
% 
%     计算中心点
%     center = mean(reshaped_data, 1);
% 
%     计算每个矩阵到中心点的距离
%     distances = sqrt(sum((reshaped_data - center).^2, 2));
% 
%     计算距离的标准差
%     std_distance = std(distances);
% 
%     设置百分之99水平的临界值
%     critical_value = norminv(0.99, 0, std_distance);

    % 进行百分之99水平的Z检验，设置权重
%     weights = ones(1, D);

    % 设置权重
%     weights( distances >= critical_value) = 0;
%     weights(distances < critical_value) = exp(-( distances(distances< critical_value)) / max(distances(distances < critical_value))/std_distance);
% 
%  weights=exp(- abs(distances-critical_value)/std_distance);
    % 将权重变形成 alpha（10，10，D）的形式
   alpha(:, :, :, small_array_idx) = repmat(reshape(weights, 1, 1, D), small_array_size, small_array_size);
   weighted_selected_array(:,:,:,small_array_idx) = selected_array .* alpha(:, :, :, small_array_idx);
end
% 初始化数组用于存储还原后的x
restored_x = zeros(M, M, D);

% 定义变量用于跟踪当前处理的方块索引
current_small_array_idx = 1;

for i = 1 : M/small_array_size
    for j = 1 : M/small_array_size
        % 添加条件判断，确保不会超过总的方块数
        if current_small_array_idx > total_small_arrays
            break; % 如果已经处理完所有方块，退出循环
        end

        % 获取第 current_small_array_idx 个方块的数据
        selected_array = weighted_selected_array(:, :, :, current_small_array_idx);

        % 计算该方块在x中的起始索引
        start_i = (i - 1) * small_array_size + 1;
        start_j = (j - 1) * small_array_size + 1;

        % 将该方块的数据复制到restored_x对应位置
        restored_x(start_i:start_i + small_array_size - 1, start_j:start_j + small_array_size - 1, :) = selected_array;

        % 更新当前处理的方块索引
        current_small_array_idx = current_small_array_idx + 1;
    end
    
    % 添加条件判断，确保不会超过总的方块数
    if current_small_array_idx > total_small_arrays
        break; % 如果已经处理完所有方块，退出外层循环
    end
end

% 现在，restored_x即为还原后的x数组
x=restored_x;