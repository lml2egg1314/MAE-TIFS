function flag = test_hill_embed(payload, iter)

    des_dir = sprintf('/data/lml/spa_test/hill_%s', num2str(payload));
    cover_dir = '/data/lml/spa_test/BB-cover-resample-256';
    stego_dir = sprintf('%s/stego-iter-%d', des_dir, iter);
    % cost_dir = sprintf('%s/cost', des_dir);
    if not(exist(stego_dir,'dir'))
        mkdir(stego_dir)
    end

    % if not(exist(cost_dir,'dir'))
    %     mkdir(cost_dir)
    % end

    parfor index = 1:20000

%         if index <= 10000
%             coverPath = [boss_cover_dir, '/', num2str(index), '.pgm'];
%         else
%             coverPath = [bows_cover_dir, '/', num2str(index-10000), '.pgm'];
%         end
        coverPath = [cover_dir, '/', num2str(index), '.pgm'];
        stegoPath = [stego_dir, '/', num2str(index), '.pgm'];
        % costPath = [cost_dir, '/', num2str(index), '.mat'];
        
    



    	%% Get embedding costs
    	% inicialization
    	cover = double(imread(coverPath));
    	wetCost = 10^8;
    	[k,l] = size(cover);
   

		% compute embedding costs \rho	
        %Get filter
        HF1 = [-1, 2, -1; 2, -4, 2; -1, 2, -1];
        H2 = fspecial('average',[3 3]);

        %% Get cost
        sizeCover = size(cover);
        padsize = max(size(HF1));
        coverPadded = padarray(cover, [padsize padsize], 'symmetric');% add padding

        R1 = conv2(coverPadded, HF1, 'same');%mirror-padded convolution
        W1 = conv2(abs(R1), H2, 'same');
        
        if mod(size(HF1, 1), 2) == 0, W1= circshift(W1, [1, 0]); end;
        if mod(size(HF1, 2), 2) == 0, W1 = circshift(W1, [0, 1]); end;

        W1 = W1(((size(W1, 1) - sizeCover(1)) / 2) + 1 : end - ((size(W1, 1) - sizeCover(1)) / 2), ((size(W1, 2) - sizeCover(2)) / 2) + 1 : end - ((size(W1, 2) - sizeCover(2)) / 2));
        rho=1 ./ (W1 + 10 ^ (-10));

        HW =  fspecial('average', [15 15]);
        cost = imfilter(rho, HW , 'symmetric', 'same');
        rho = cost;




    	% adjust embedding costs
    	rho(rho > wetCost) = wetCost; % threshold on the costs
    	rho(isnan(rho)) = wetCost; % if all xi{} are zero threshold the cost
    	rhoP1 = rho;
    	rhoM1 = rho;
    	rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
    	rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
    

    	%% Embedding simulator
    	stego = EmbeddingSimulator(cover, rhoP1, rhoM1, payload*numel(cover), false);
    	stego = uint8(stego);
    
    
    	%% save stego and cost
    	imwrite(stego, stegoPath);

    	% save_cost(rhoP1, rhoM1, costPath);

    end



    	flag = 'Finish';

end


function save_cost(rhoP1, rhoM1, costPath)
  	save(costPath, 'rhoP1', 'rhoM1');
end



%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound). 
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)

    n = numel(x);   
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    if fixEmbeddingChanges == 1
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
    else
        RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    end
    randChange = rand(size(x));
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;
    
    function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            m3 = ternary_entropyf(pP1, pM1);
            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                return;
            end
        end        
        
        l1 = 0; 
        m1 = double(n);        
        lambda = 0;
        
        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
            lambda = l1+(l3-l1)/2; 
            pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            m2 = ternary_entropyf(pP1, pM1);
    		if m2 < message_length
    			l3 = lambda;
    			m3 = m2;
            else
    			l1 = lambda;
    			m1 = m2;
            end
    		iterations = iterations + 1;
        end
    end
    
    function Ht = ternary_entropyf(pP1, pM1)
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end
end
