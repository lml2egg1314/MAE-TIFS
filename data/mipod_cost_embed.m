function flag = mipod_cost_embed(cover_dir, stego_dir, cost_dir, prob_dir, payload)
    
    if not(exist(stego_dir,'dir'))
        mkdir(stego_dir)
    end

    if not(exist(cost_dir,'dir'))
        mkdir(cost_dir)
    end
    
    if not(exist(prob_dir,'dir'))
        mkdir(prob_dir)
    end
    
    
    parfor index = 1:20000

%         if index <= 10000
%             cover_path = [BossbassDir, '/', num2str(index), '.pgm'];
%         else
%             cover_path = [BowsDir, '/', num2str(index-10000), '.pgm'];
%         end
        cover_path = [cover_dir, '/', num2str(index), '.pgm'];
        stego_path = [stego_dir, '/', num2str(index), '.pgm'];
        cost_path = [cost_dir, '/', num2str(index), '.mat'];
        prob_path = [prob_dir, '/', num2str(index), '.mat'];
        
    



    	%% Get embedding costs
    	% inicialization
    	cover = double(imread(cover_path));
        
        % Compute Variance and do the flooring for numerical stability
        WienerResidual = cover - wiener2(cover,[2,2]);
        Variance = VarianceEstimationDCT2D(WienerResidual,9,9);
        Variance(Variance< 0.01) = 0.01;

        % Compute Fisher information and smooth it
        FisherInformation = 1./Variance.^2;
        FisherInformation = imfilter(FisherInformation,fspecial('average',7),'symmetric');

        % Compute embedding change probabilities and execute embedding
        FI = FisherInformation(:)';

        % Ternary embedding change probabilities
        beta = TernaryProbs(FI,payload);

        % Simulate embedding
        Stego = cover;
        beta = 2 * beta;
        r = rand(1,numel(cover));
        ModifPM1 = (r < beta);                % Cover elements to be modified by +-1
        r = rand(1,numel(cover));
        Stego(ModifPM1) = cover(ModifPM1) + 2*(round(r(ModifPM1))) - 1; % Modifying X by +-1
        Stego(Stego>255) = 253;                    % Taking care of boundary cases
        Stego(Stego<0)   = 2;
        prob_map = reshape(beta,size(cover));
        
        % calculate rho
        wetCost = 10^8;
        pChange = reshape(beta/2,size(cover));
        rho = log((1 ./ pChange) - 2);
        rho(rho > wetCost) = wetCost; % threshold on the costs
        rho(isnan(rho)) = wetCost; % if all xi{} are zero threshold the cost
        rhoP1 = rho;
        rhoM1 = rho;
        rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
        rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value


    	
    
    	%% save stego and cost
        stego = uint8(Stego);
    	imwrite(stego, stego_path);

    	save_cost(rhoP1, rhoM1, cost_path);
        save_prob(prob_map, prob_path);

    end



    flag = 'Finish';

end





% Estimation of the pixels' variance based on a 2D-DCT (trigonometric polynomial) model
function EstimatedVariance = VarianceEstimationDCT2D(Image, BlockSize, Degree)
    % verifying the integrity of input arguments
    if ~mod(BlockSize,2)
        error('The block dimensions should be odd!!');
    end
    if (Degree > BlockSize)
        error('Number of basis vectors exceeds block dimension!!');
    end

    % number of parameters per block
    q = Degree*(Degree+1)/2;

    % Build G matirx
    BaseMat = zeros(BlockSize);BaseMat(1,1) = 1;
    G = zeros(BlockSize^2,q);
    k = 1;
    for xShift = 1 : Degree
        for yShift = 1 : (Degree - xShift + 1)
            G(:,k) = reshape(idct2(circshift(BaseMat,[xShift-1 yShift-1])),BlockSize^2,1);
            k=k+1;
        end
    end

    % Estimate the variance
    PadSize = floor(BlockSize/2*[1 1]);
    I2C = im2col(padarray(Image,PadSize,'symmetric'),BlockSize*[1 1]);
    PGorth = eye(BlockSize^2) - (G*((G'*G)\G'));
    EstimatedVariance = reshape(sum(( PGorth * I2C ).^2)/(BlockSize^2 - q),size(Image));
end




% Computing the embedding change probabilities
function [beta] = TernaryProbs(FI,alpha)

    load('mipod_ixlnx3.mat');

    % Absolute payload in nats
    payload = alpha * length(FI) * log(2);

    % Initial search interval for lambda
    [L, R] = deal (10^3, 10^6);

    fL = h_tern(1./invxlnx3_fast(L*FI,ixlnx3)) - payload;
    fR = h_tern(1./invxlnx3_fast(R*FI,ixlnx3)) - payload;
    % If the range [L,R] does not cover alpha enlarge the search interval
    while fL*fR > 0
        if fL > 0
            R = 2*R;
            fR = h_tern(1./invxlnx3_fast(R*FI,ixlnx3)) - payload;
        else
            L = L/2;
            fL = h_tern(1./invxlnx3_fast(L*FI,ixlnx3)) - payload;
        end
    end

    % Search for the labmda in the specified interval
    [i, fM, TM] = deal(0, 1, zeros(60,2));
    while (abs(fM)>0.0001 && i<60)
        M = (L+R)/2;
        fM = h_tern(1./invxlnx3_fast(M*FI,ixlnx3)) - payload;
        if fL*fM < 0, R = M; fR = fM;
        else          L = M; fL = fM; end
        i = i + 1;
        TM(i,:) = [fM,M];
    end
    if (i==60)
        M = TM(find(abs(TM(:,1)) == min(abs(TM(:,1))),1,'first'),2);
    end
    % Compute beta using the found lambda
    beta = 1./invxlnx3_fast(M*FI,ixlnx3);

end



% Fast solver of y = x*log(x-2) paralellized over all pixels
function x = invxlnx3_fast(y,f)

    i_large = y>1000;
    i_small = y<=1000;

    iyL = floor(y(i_small)/0.01)+1;
    iyR = iyL + 1;
    iyR(iyR>100001) = 100001;

    x = zeros(size(y));
    x(i_small) = f(iyL) + (y(i_small)-(iyL-1)*0.01).*(f(iyR)-f(iyL));

    z = y(i_large)./log(y(i_large)-2);
    for j = 1 : 20
        z = y(i_large)./log(z-2);
    end
    x(i_large) = z;

end
        

% Ternary entropy function expressed in nats
function Ht = h_tern(Probs)

p0 = 1-2*Probs;
P = [p0(:);Probs(:);Probs(:)];
H = -(P .* log(P));
H((P<eps)) = 0;
Ht = nansum(H);

end

    	


function save_cost(rhoP1, rhoM1, costPath)
  	save(costPath, 'rhoP1', 'rhoM1');
end

function save_prob(prob_map, probPath)
    save(probPath, 'prob_map');
end


