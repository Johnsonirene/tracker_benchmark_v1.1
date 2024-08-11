function [wf,hf] = admm_base(xf, xcf,  xf_p, wf_p, use_sz, params, yf, small_filter_sz)


    % feature size
    sz = size(xf);
    T = prod(use_sz);

    Sxy = bsxfun(@times, xf, conj(yf));
    Sxx = bsxfun(@times, xf, conj(xf));
    x_fuse = xf + xf_p;
    Sxx_fuse = bsxfun(@times, x_fuse, conj(x_fuse));

    % initialize h
    hf = single(zeros(sz));

    % initialize Lagrangian multiplier
    zetaf = single(zeros(sz));

    % penalty
    mu = params.mu;
    beta = params.beta;
    mu_max = 10000;
    lambda = params.t_lambda;
    gamma = params.gamma;
    
    i = 1;
    % ADMM iterations
    while (i <= params.admm_iterations)
        % Solve for G (please refer to the paper for more details)
%         g_f = bsxfun(@rdivide,(1/T) * Sxy + mu * hf - zetaf, (1/T) * Sxx + (params.yta/T) * Sxx_cf + mu);
        wf = (Sxy + gamma * (wf_p .* Sxx_fuse) + mu * hf - zetaf) ./ (Sxx + gamma * Sxx_fuse + mu);


        % Solve for H
        h = (ifft2(mu * wf + zetaf)) ./ (lambda/T + mu);
%         hf = fft2(bsxfun(@rdivide, ifft2((mu*g_f) + zetaf), (T/((mu*T)+ params.admm_lambda))));
        [sx, sy, h] = get_subwindow_no_window(h, floor(use_sz/2), small_filter_sz);
        t = single(zeros(use_sz(1), use_sz(2), size(h, 3)));
        t(sx, sy, :) = h;
        hf = fft2(t);

        % Update L
        zetaf = zetaf + (mu * (wf - hf));

        % Update mu - betha = 10
        mu = min(beta * mu, mu_max);
        i = i + 1;
    end

end
