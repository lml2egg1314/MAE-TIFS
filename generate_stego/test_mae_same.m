 
 %% dir
 steganography_set = {'hill','cmd_hill','suni','mipod','wow'};
 
 for ln = 1:3
     params.start = 1;
     params.listNum = ln;
     %  for i =1 % for server-197
     % for i = 2:3 % for server-202
     for i = 4:5 % for server-201
        steganography = steganography_set{i};
        for j = 1:4
           
            payload = j/10;
            sp_dir = sprintf('%s_%s', steganography, num2str(payload));
            params.sp_dir = sp_dir;
            base_dir = sprintf('/data/lml/spa_test/%s', sp_dir);
            grad_dir = sprintf('%s/%s', base_dir, num2str(params.listNum));
            postfix = 'resample-256';
            

            output_dir = sprintf('%s/output_same', grad_dir);

            params.payload = payload;
            params.steganography = steganography;
            params.IMAGE_SIZE = 256;
     %%
            params.cover_dir = sprintf('/data/lml/spa_test/BB-cover-%s', postfix);
            params.stego_dir = sprintf('%s/stego', base_dir);

            params.cost_dir = sprintf('%s/cost', base_dir);
           
            params.dnet_grad_dir = sprintf('%s/mm_grad_10', grad_dir);
            %for xunet pretrained
%             params.cover_xu_grad_dir = sprintf('%s/cover-xu-grad', base_dir);
%             params.msgs_dir = sprintf('%s/msgs_xu_%d', base_dir, 5);
            % params.cover_srnet_grad_dir = sprintf('%s/cover-srnet-gard', base_dir);

           
           params.PARA = 1;
%             params.listNum = 1;

            % params.p1 = 0;
            params.p0 = 0.1;
            params.p1 = 2;
            params.p2 = 2;
            params.p3 = 0.7;

            %% mode = 1
            params.mode1 = 1;
            params.mode2 = 1;
            params.mode3 = 1;
            % params.ss_number = 5;
            for k = 5
                
                params.ss_number = k;
                params.output_stego_dir = sprintf('%s/stego', output_dir);
                params.output_cost_dir = sprintf('%s/cost', output_dir);
              
                disp(params);
                tic;
                flag = mae_same(params);
           
                toc;
                srm_and_ensemble_mae;
            end
        end
     end

 end