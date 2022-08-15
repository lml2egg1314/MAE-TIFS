% payload = 0.4;

for iter = 0
    for i = 1:4
        payload = i/10;
        tic;
        if iter == 0
            des_dir = sprintf('/data/lml/spa_test/wow_%s', num2str(payload));
            cover_dir = '/data/lml/spa_test/BB-cover-resample-256';
            stego_dir = sprintf('%s/stego', des_dir);
            cost_dir = sprintf('%s/cost', des_dir);
            flag = wow_cost_embed(cover_dir, stego_dir, cost_dir, payload);
        else 
            des_dir = sprintf('/data/lml/spa_test/wow_%s', num2str(payload));
            cover_dir = '/data/lml/spa_test/BB-cover-resample-256';
            stego_dir = sprintf('%s/stego-iter-%d', des_dir, iter);
            flag = wow_embed(cover_dir, stego_dir, payload);
        end
        toc;
    end
end
exit;