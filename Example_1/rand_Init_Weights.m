function W = rand_Init_Weights(r, c)
%Random initialization weights
W = zeros(r, c);

epsilon_init = 0.10;
W=rand(r, c) * 2 * epsilon_init - epsilon_init;
end
