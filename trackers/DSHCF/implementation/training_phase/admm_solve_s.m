function s = admm_solve_s(params, use_sz, model_s, hf)


    mu_max = 10000;
    beta = 10;
    T = prod(use_sz);
    mu = 3;
    lambda = params.t_lambda;
    lambda2 = params.t_lambda2;
%     alphas = params.alphas;
    h_w = T*real(ifft2(hf));
    hw = h_w;
    Hh = sum(hw.^2,3);
    w =params.s_init*single(ones(use_sz));
    q = w;
    m = w;

    i = 1;
    while (i <= params.admm_iterations)
        %   solve for w- please refer to the paper for more details
      %  w = (q-m)/(1+(params.admm_lambda1/mu)*Hh);
        s = bsxfun(@rdivide, (q-m), (1 + (lambda/mu) * Hh));
        %   solve for q
        q = (lambda2 * model_s + mu * (s+m))/(lambda2 + mu);
        
        %   update m
        m = m + mu * (s - q);
        
        %   update mu- betha = 10.
        mu = min(beta * mu, mu_max);
        i = i+1;
    end
%   model_s = alphas * s + (1-alphas) * model_s;

end
