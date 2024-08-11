function n = zuidagongyueshu(a)
    n = 1;
    for i = 2 : a-1
        if (rem(a,i) == 0)
            n = i;
        end
    end
end
