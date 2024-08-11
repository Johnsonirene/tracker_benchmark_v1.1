alpha=admm_alpha(f_f,xlf{k1},yf{k},lamda2,W,alpha)
Tk=xlf{k1}.*xlf{k1};
tk=(real(ifft2(conj(Tk) .* Tk - conj(Tk).*yf{k})))
for j=1:size(xlf{k1},3)
    ttk=tk(:,:,j);
    fk = reshape(sum(real(ttk))), 1, []);
    [maxw ,index] =sort(W(j,:),'descend');
        p_ai= sum((2*lamda2*maxw(1,1)));
        pp_ai= p_ai.*alpha(:, :, index(1));
        alpha=( pp_ai+ fk)/ p_ai;
end