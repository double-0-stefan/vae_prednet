image = zeros(3,32,32);
ii =image(:);

cov = zeros(length(ii),length(ii));
neighbourhood = 3;

for a1 = 1:32
    for b1 = 1:32
        for c1 = 1:3
            for a = 1:32
                for b = 1:32
                    for c = 1:3
                        % goes direct to chol
                        
                        if abs(a-a1) <= neighbourhood && abs(b-b1) <= neighbourhood %&& c==c1
                            index = (c-1)*32*32+(b-1)*32+a;
                            index1 = (c1-1)*32*32+(b1-1)*32+a1;
                            if index >= index1
                                cov(index,index1) = 1;
                            end
                        end
                    end
                end
            end
        end
    end
end
%%
cov = cov + 10*eye(length(ii));
%%
spy(cov)
%%
R=chol(cov);
%%
cc = cov*cov';
%%
spy(cc)