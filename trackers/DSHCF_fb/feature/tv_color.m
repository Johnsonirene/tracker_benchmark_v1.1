function J = tv_color(I, iter, dt, ep, lam, I0, C)
% Total Variation denoising for color images.
% Example: J = tv_color(I, iter, dt, ep, lam, I0)
% Input: 
%    I    - image (double array, RGB 0-1),
%    iter - number of iterations,
%    dt   - time step [0.2],
%    ep   - epsilon (of gradient regularization) [1],
%    lam  - fidelity term lambda [0],
%    I0   - input (noisy) image [I0=I]
%    C    - constant added to the fidelity term (optional) [0]

I = im2double(I); % 转换为 double 类型
if ~exist('I0', 'var') || isempty(I0)
    I0 = I; % 默认情况下，将 I0 设置为输入图像 I
else
    I0 = im2double(I0); % 确保 I0 也是 double 类型
end

% Set default values if not provided
if ~exist('ep', 'var'), ep = 1; end
if ~exist('dt', 'var'), dt = ep / 5; end  % dt below the CFL bound
if ~exist('lam', 'var'), lam = 0; end
if ~exist('I0', 'var'), I0 = I; end
if ~exist('C', 'var'), C = 0; end

[ny, nx, nc] = size(I); % nc is the number of channels (3 for RGB)
J = zeros(size(I));     % Initialize output image

for ch = 1:nc  % Loop over each channel
    J(:,:,ch) = tv(I(:,:,ch), iter, dt, ep, lam, I0(:,:,ch), C);
end

end

function J = tv(I, iter, dt, ep, lam, I0, C)
% Process a single channel using TV denoising
ep2 = ep^2;
[ny, nx] = size(I);

for i = 1:iter  % do iterations
    % estimate derivatives
    I_x = (I(:, [2:nx, nx]) - I(:, [1, 1:nx-1])) / 2;
    I_y = (I([2:ny, ny], :) - I([1, 1:ny-1], :)) / 2;
    I_xx = I(:, [2:nx, nx]) + I(:, [1, 1:nx-1]) - 2 * I;
    I_yy = I([2:ny, ny], :) + I([1, 1:ny-1], :) - 2 * I;
    Dp = I([2:ny, ny], [2:nx, nx]) + I([1, 1:ny-1], [1, 1:nx-1]);
    Dm = I([1, 1:ny-1], [2:nx, nx]) + I([2:ny, ny], [1, 1:nx-1]);
    I_xy = (Dp - Dm) / 4;

    % compute flow
    Num = I_xx .* (ep2 + I_y.^2) - 2 * I_x .* I_y .* I_xy + I_yy .* (ep2 + I_x.^2);
    Den = (ep2 + I_x.^2 + I_y.^2).^(3/2);
    I_t = Num ./ Den + lam .* (I0 - I + C);

    I = I + dt * I_t;  % evolve image by dt
end

J = I;
end
