
img = imread('D:\ML\Visual_tracking\benchmark\drone-tracking-master\experiments\trackers\DSHCF_fb\seq\Yacht2\00001.jpg');

Vy=0;
Vx=0;
center=pos+[Vy Vx];
pixel_template=get_pixels(img, center, round(sz*currentScaleFactor), sz); 
lambda = 1;  % 正则化参数
rho = 0.01;     % ADMM参数
num_iterations = 10;  % 迭代次数

denoised_img = admm_denoise_color(pixel_template, lambda, rho, num_iterations);

figure;
subplot(1,2,1);
imshow(pixel_template);
title('Original Image');

subplot(1,2,2);
imshow(denoised_img);
title('Denoised Image using ADMM');