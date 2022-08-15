function flag = suni_embed(cover_dir, stego_dir, payload)
    
    if not(exist(stego_dir,'dir'))
        mkdir(stego_dir)
    end

%     if not(exist(cost_dir,'dir'))
%         mkdir(cost_dir)
%     end

    for index = 1:20000

%         if index <= 10000
%             coverPath = [BossbassDir, '/', num2str(index), '.pgm'];
%         else
%             coverPath = [BowsDir, '/', num2str(index-10000), '.pgm'];
%         end
        cover_path = [cover_dir, '/', num2str(index), '.pgm'];
        
        stego_path = [stego_dir, '/', num2str(index), '.pgm'];
%         cost_path = [cost_dir, '/', num2str(index), '.mat'];
        




        %% Get embedding costs
        sgm = 1;

        %% Get 2D wavelet filters - Daubechies 8
        % 1D high pass decomposition filter
        hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, ...
                -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768];
        % 1D low pass decomposition filter
        lpdf = (-1).^(0:numel(hpdf)-1).*fliplr(hpdf);
        % construction of 2D wavelet filters
        F{1} = lpdf'*hpdf;
        F{2} = hpdf'*lpdf;
        F{3} = hpdf'*hpdf;

        %% Get embedding costs
        % inicialization
        cover = double(imread(cover_path));
        wetCost = 10^8;
        [k,l] = size(cover);

        % add padding
        padSize = max([size(F{1})'; size(F{2})'; size(F{3})']);
        coverPadded = padarray(cover, [padSize padSize], 'symmetric');

        xi = cell(3, 1);
        for fIndex = 1:3
            % compute residual
            R = conv2(coverPadded, F{fIndex}, 'same');
            % compute suitability
            xi{fIndex} = conv2(1./(abs(R)+sgm), rot90(abs(F{fIndex}), 2), 'same');
            % correct the suitability shift if filter size is even
            if mod(size(F{fIndex}, 1), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [1, 0]); end;
            if mod(size(F{fIndex}, 2), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [0, 1]); end;
            % remove padding
            xi{fIndex} = xi{fIndex}(((size(xi{fIndex}, 1)-k)/2)+1:end-((size(xi{fIndex}, 1)-k)/2), ((size(xi{fIndex}, 2)-l)/2)+1:end-((size(xi{fIndex}, 2)-l)/2));
        end

        % compute embedding costs \rho
        rho = xi{1} + xi{2} + xi{3};

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
    	imwrite(stego, stego_path);

%     	save_cost(rhoP1, rhoM1, cost_path);

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
