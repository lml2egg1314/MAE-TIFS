

%parpool(8);

payload = params.payload;
steganography = params.steganography;

file_num = 20000;



% for i = 1:length(steganography_set)

%   steganography = steganography_set{i};
    % params.p0 = 0;
    % params.p1 = 0;
    % params.p2 = 0;
    % params.p3 = 0;
    % params.mode2 = 1;

  extract_and_classify(steganography, payload, file_num, params);

% end





function extract_and_classify(steganography, payload, file_num, params)

  
  cover_dir = params.cover_dir;
  
  stego_dir = params.output_stego_dir;

  dir_exist = exist(stego_dir, 'dir');
  if dir_exist ~= 7

    fprintf('%s does not exist \n', stego_dir);
    fprintf('------------------------- \n');

    return
  end

  total_start = tic;




  dataset_division_array_path = './dataset_division_array.mat';
  dataset_division_array_mat = load(dataset_division_array_path);
  dataset_division_array = dataset_division_array_mat.dataset_division_array;

  division_num = params.listNum;
  dataset_division = dataset_division_array{division_num};
  training_set = dataset_division.training_set;
  test_set = dataset_division.test_set;





  cover_set = {};
  stego_set = {};

  for index = 1 : 5000
    cover_set{end + 1} = sprintf('%s/%d.pgm', cover_dir, training_set(index));
    stego_set{end + 1} = sprintf('%s/%d.pgm', stego_dir, training_set(index));
  end
  for index = 1 : 5000
    cover_set{end + 1} = sprintf('%s/%d.pgm', cover_dir, test_set(index));
    stego_set{end + 1} = sprintf('%s/%d.pgm', stego_dir, test_set(index));
  end
  

  fprintf('Extract cover SRM features ----- \n');
  srm_features_path = sprintf('/data/lml/spa_test/srm_cover%s.mat', num2str(params.listNum));
  if params.start == 0
    cover_features = srm_extract(cover_set);
    
    save(srm_features_path, 'cover_features', '-v7.3');
  else
    features_mat = load(srm_features_path);
    cover_features = features_mat.cover_features;
  end
%   features = cover_features;
%   save('/data/lml/stt/hill_0.4/srm_cover1.mat', 'cover_features', '-v7.3');

  fprintf('Steganography: %s ----- \n\n', steganography);
  fprintf('Extract stego SRM features ----- \n');
  stego_features = srm_extract(stego_set);
%   save('/data/lml/stt/hill_0.4/srm_stego1.mat', 'stego_features', '-v7.3');





  test_acc = ensemble_classify(cover_features, stego_features, training_set, test_set);

  total_end = toc(total_start);
  
  
  
  fprintf('SRM and ensemble results ----- \n')
  fprintf('Test accuracy for # dnet-ms-%s-%s #: %.4f \n', params.sp_dir, num2str(params.listNum), test_acc);

  file_id = fopen('acc_log_new.txt','a');
  fprintf(file_id,'%s  dnet-mae-n-%d-%s-%s: %.4f\n', datestr(now), params.ss_number, params.sp_dir, num2str(params.listNum), test_acc);
  fclose(file_id);

  fprintf('Total time: %.2f seconds. \n', total_end);
  fprintf('------------------------- \n')

end




function [features] = srm_extract(image_set)

  srm_start = tic;

  file_num = length(image_set);
  feature_num = 34671;

  features = zeros(file_num, feature_num);
  parfor i = 1:file_num
    image_item = image_set(i);
    feature_item = SRM(image_item);

    feature_item = struct2cell(feature_item);
    feature_item = [feature_item{:}];

    features(i, :) = feature_item;
  end

  srm_end = toc(srm_start);

  fprintf('SRM extracted %d images in %.2f seconds, in average %.2f seconds per image. \n\n', numel(image_set), srm_end, srm_end / numel(image_set));


end


function [test_acc] = ensemble_classify(cover_features, stego_features, training_set, testing_set)

  train_cover = cover_features(1:5000, :);
  train_stego = stego_features(1:5000, :);

  test_cover = cover_features(5001:10000, :);
  test_stego = stego_features(5001:10000, :);

  settings = struct('verbose', 2);

  train_start = tic;

  fprintf('Ensemble train start ----- \n');

  [trained_ensemble,results] = ensemble_training(train_cover, train_stego, settings);

  train_end = toc(train_start);


  fprintf('\n');

  test_start = tic;

  fprintf('Ensemble test start ----- \n');

  test_results_cover = ensemble_testing(test_cover, trained_ensemble);
  test_results_stego = ensemble_testing(test_stego, trained_ensemble);

  test_end = toc(test_start);


  % Predictions: -1 stands for cover, +1 for stego
  false_alarms = sum(test_results_cover.predictions ~= -1);
  missed_detections = sum(test_results_stego.predictions ~= +1);

  num_testing_samples = size(test_cover, 1) + size(test_stego, 1);

  testing_error = (false_alarms + missed_detections) / num_testing_samples;

  fprintf('Train time: %.2f seconds, Test time: %.2f seconds. \n\n', train_end, test_end);

  test_acc = 1 - testing_error;

end

