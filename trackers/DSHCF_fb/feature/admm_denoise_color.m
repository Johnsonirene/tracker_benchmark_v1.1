function denoised_img = admm_denoise_color(img, lambda, rho, num_iterations)
    img = double(img);
    [rows, cols, channels] = size(img);

    % 初始化
    u = img;  % 去噪图像初始化为原图
    z = zeros(rows, cols, channels, 2);  % 梯度的辅助变量
    b = zeros(rows, cols, channels, 2);  % 对偶变量

    [Dx, Dy] = createGradientOperators(rows, cols);

for i = 1:num_iterations
    for ch = 1:channels
        % 重塑图像通道为向量
        u_vec = reshape(u(:, :, ch), [], 1);

        % 梯度和散度的计算需要正确的向量形式
        ux_vec = Dx * u_vec;
        uy_vec = Dy * u_vec;

        % 重塑为图像形式进行更新
        ux = reshape(ux_vec, rows, cols);
        uy = reshape(uy_vec, rows, cols);

        % z-update（使用软阈值处理）
        z(:, :, ch, 1) = soft_threshold(ux + b(:, :, ch, 1), lambda / rho);
        z(:, :, ch, 2) = soft_threshold(uy + b(:, :, ch, 2), lambda / rho);

        % 更新 b
        b(:, :, ch, 1) = b(:, :, ch, 1) + (ux - z(:, :, ch, 1));
        b(:, :, ch, 2) = b(:, :, ch, 2) + (uy - z(:, :, ch, 2));

        % 更新 u 使用适当重塑的 z 和 b
        u(:, :, ch) = reshape(u_vec - rho * (Dx' * reshape(z(:, :, ch, 1) - b(:, :, ch, 1), [], 1) + Dy' * reshape(z(:, :, ch, 2) - b(:, :, ch, 2), [], 1)), rows, cols);
    end
end
    denoised_img = uint8(u);
end

function z = soft_threshold(x, kappa)
    z = max(0, 1-kappa./abs(x)) .* x;
end

function [Dx, Dy] = createGradientOperators(m, n)
    % 构建水平梯度操作符
    Dx = spdiags([-ones(m, 1) ones(m, 1)], [0 1], m, m);
    Dx(:, end) = 0;  % 防止边界溢出
    Dx = kron(speye(n), Dx);

    % 构建垂直梯度操作符
    Dy = spdiags([-ones(n, 1) ones(n, 1)], [0 1], n, n);
    Dy(:, end) = 0;  % 防止边界溢出
    Dy = kron(Dy, speye(m));
end
