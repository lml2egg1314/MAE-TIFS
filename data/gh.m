% payload = 0.4;

for iter = 0:10
    for i = 1:4
        payload = i/10;
        tic;
        if iter == 0
            des_dir = sprintf('/data/lml/spa_test/suni_%s', num2str(payload));
            cover_dir = '/data/lml/spa_test/BB-cover-resample-256';
            stego_dir = sprintf('%s/stego', des_dir);
            cost_dir = sprintf('%s/cost', des_dir)
            flag = hill_cost_embed(payload);
        else 
            des_dir = sprintf('/data/lml/spa_test/suni_%s', num2str(payload));
            cover_dir = '/data/lml/spa_test/BB-cover-resample-256';
            stego_dir = sprintf('%s/stego-iter-%d', des_dir, iter);
            flag = test_hill_embed(payload, iter);
        end
        toc;
    end
end
exit;