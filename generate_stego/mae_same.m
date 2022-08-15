function flag = mae_same(params)
    
    %% load parameters
    
     % for hyper parameters
    IMAGE_SIZE = params.IMAGE_SIZE;
    payload = params.payload;
    listNum = params.listNum;
    ss_number = params.ss_number;
    
    %     for dir
    cover_dir = params.cover_dir;
    dnet_grad_dir = params.dnet_grad_dir;
%     cover_srnet_grad_dir = params.cover_srnet_grad_dir;
    cost_dir = params.cost_dir;
    output_stego_dir = params.output_stego_dir;
    output_cost_dir = params.output_cost_dir;
  
    if not(exist(output_stego_dir,'dir'))
        mkdir(output_stego_dir)
    end

    if not(exist(output_cost_dir,'dir'))
        mkdir(output_cost_dir)
    end
   

   
    %% load test index list
    indexListPath = ['./index_list/', num2str(listNum), '/test_list.mat'];
    IndexList = load(indexListPath);
    index_list = IndexList.index;
    len = length(index_list);
    
    %len = 100;
     %% haper-parameter
%     IMAGE_SIZE = params.IMAGE_SIZE;
    p0 = params.p0;
    p2 = params.p2;
    p3 = params.p3;
    rates = zeros(len,3);
    parfor index_it = 1:len
%         total_start = tic;
    
        index = index_list(index_it);

        %% load data    
        cover_path = [cover_dir, '/', num2str(index), '.pgm'];
        
        %cg: cover_grad; sg:stego_grad
        mm_path = [dnet_grad_dir, '/', num2str(index), '.mat'];
        cost_path = [cost_dir, '/', num2str(index), '.mat'];
        
        output_stego_path = [output_stego_dir, '/', num2str(index), '.pgm'];
        output_cost_path = [output_cost_dir, '/', num2str(index), '.mat'];
        
        cover = double(imread(cover_path));
        [pre_rhoP1, pre_rhoM1] = load_cost(cost_path);
        rhoP1 = pre_rhoP1;
        rhoM1 = pre_rhoM1;
        
        mm_grad = load_grad(mm_path);
        grad = sign(mm_grad(1:ss_number+1, :, :));
        
       
        cgv = squeeze(mm_grad(1,:,:));
      
             %% get max p ratio cover grad
        acgv = abs(cgv(:));
        acgv = sort(acgv, 'descend');
        p_value = acgv(round(IMAGE_SIZE * IMAGE_SIZE * p0));

        max_cgv = (abs(cgv) > p_value);
         %% 
        
        srho = sort(rhoP1(:));
        r_value = srho(round(IMAGE_SIZE * IMAGE_SIZE * p3));

        min_rho = (rhoP1 < r_value);
        
        sum_sgs = squeeze(sum(grad,1)); %ndims:; 3
        mp_same = (sum_sgs == ss_number + 1);
        mm_same = (sum_sgs == -(ss_number + 1));
        
        r1 = (sum(mp_same(:))+sum(mm_same(:)))/(256*256);
        
        modulation_map = min_rho .* max_cgv;
        r2 = sum(modulation_map(:))/(256*256);
        mp_all = mp_same .* modulation_map;
        mm_all = mm_same .* modulation_map;
        rhoP1 = rhoP1 + mp_all * p2 ;
        rhoM1 = rhoM1 + mm_all * p2 ;
%         modulation_map = (abs(sum_sgs) == ss_number) .* min_rho .* max_cgv
        
        r3 = (sum(mp_all(:))+ sum(mm_all(:)))/(256*256);  
        rates(index_it,:) = [r1, r2, r3];

        
        %% Get embedding costs & stego
        % inicialization
        wetCost = 10^8;

        % adjust embedding costs
        rhoP1(rhoP1 > wetCost) = wetCost; % threshold on the costs
        rhoM1(rhoM1 > wetCost) = wetCost;
        rhoP1(isnan(rhoP1)) = wetCost; % if all xi{} are zero threshold the cost
        rhoM1(isnan(rhoM1)) = wetCost;
        rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
        rhoM1(cover==0) = wetCost;

        stego = EmbeddingSimulator(cover, rhoP1, rhoM1, payload*numel(cover), false);
        stego = uint8(stego);

        imwrite(stego, output_stego_path);

        save_cost(rhoP1, rhoM1, output_cost_path);
%         total_end = toc(total_start);
               
    end
    mean_rate = mean(rates)
    file_id = fopen('rates_log_new.txt','a');
    fprintf(file_id,'%s  dnet-mae-rate-n-%d-%s-%s: %.4f; :%.4f; %.4f\n', datestr(now), params.ss_number, params.sp_dir, num2str(params.listNum), mean_rate(1),mean_rate(2),mean_rate(3));
    fclose(file_id);
 
    flag = 'Finish';

end


function save_cost(best_cost_p1, best_cost_m1, costPath)
    
    rhoP1 = best_cost_p1;
    rhoM1 = best_cost_m1;
    save(costPath, 'rhoP1', 'rhoM1');
    
end

% function msgs = load_msgs(msgs_path)
%     tmp = load(msgs_path);   
%     msgs = tmp.msgs;
% end

function [pre_rhoP1, pre_rhoM1] = load_cost(preCostPath)

    Pre_Rho = load(preCostPath);
    pre_rhoP1 = Pre_Rho.rhoP1;
    pre_rhoM1 = Pre_Rho.rhoM1;

end

% function [sign_grad, allgrad] = load_grad(preGradPath)
% 
%     Grad = load(preGradPath);
%     sign_grad = Grad.sign_grad;
%     allgrad = Grad.grad;
% 
% end

function [allgrad] = load_grad(preGradPath)

    Grad = load(preGradPath);
    
    allgrad = Grad.mm_grad;

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
