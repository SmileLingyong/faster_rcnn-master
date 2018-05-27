function save_model_path = proposal_train(conf, imdb_train, roidb_train, varargin)
% save_model_path = proposal_train(conf, imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
%% -------------------- CONFIG --------------------
% inputs
    ip = inputParser;
    ip.addRequired('conf',                                      @isstruct);
    ip.addRequired('imdb_train',                                @iscell);
    ip.addRequired('roidb_train',                               @iscell);
    ip.addParamValue('do_val',              false,              @isscalar);
    ip.addParamValue('imdb_val',            struct(),           @isstruct);
    ip.addParamValue('roidb_val',           struct(),           @isstruct);
    
    ip.addParamValue('val_iters',           500,                @isscalar);
    ip.addParamValue('val_interval',        2000,               @isscalar);
    ip.addParamValue('snapshot_interval',...
                                            10000,              @isscalar);
                                                                       
    % Max pixel size of a scaled input image
    ip.addParamValue('solver_def_file',     fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'solver.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('net_file',            fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',          'Zeiler_conv5', ...
                                                        @isstr);
    
    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    
%% try to find trained model
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', opts.cache_name, imdbs_name);
    save_model_path = fullfile(cache_dir, 'final');
    if exist(save_model_path, 'file')
        return;
    end
    
%% init  
    % init caffe solver
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', opts.cache_name, imdbs_name);
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_solver = caffe.Solver(opts.solver_def_file);
    caffe_solver.net.copy_from(opts.net_file);
    
    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
    diary(log_file);   
    
    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);
    
    % set gpu/cpu
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    
    disp('conf:');
    disp(conf);
    disp('opts:');
    disp(opts);
    
%% making tran/val data
    fprintf('Preparing training data...');
    [image_roidb_train, bbox_means, bbox_stds]...
                            = proposal_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train);
    % 将所有图片imdbs以及roidbs结构体数据传入，构造image_roidb_train结构体数据，
    % 这里我们只用了每张图片rois结构体数据中的ground truth，并没有用ss产生的rois(region proposals)。 
    % 然后使用Anchors机制计算每张图片产生的anchors，结合每张图片的rois结构体中的ground truth计算overlap，
    % 接着通过分析找到这些anchors中哪些属于前景，哪些属于背景，并计算前景边框回归到其所属ground truth的平移缩放参数，加上ground truth类别标签构成bbox_targets参数，之后又将前景的类别都设置为1了，背景则依然是0
    % 然后计算所有图片的bbox_targets的均值和方差，并将bbox_targets用该均值和方差做归一化处理。 
    % image_roidb_train   : 10022x1 struct array with : 该参数主要理解bbox_targets的含义。
    %   image_path
    %   image_id
    %   im_size
    %   imdb_name
    %   num_classes
    %   boxes             : [ground truth] 这里只保留了每张图片的ground truth
    %   class
    %   image
    %   bbox_targets      : 每个前景rois与其gt_rois进行bounding boxes regression需要做的平移尺度变换(4个参数)。 再加上其所属ground truth类别标签(1参数)，构成bbox_targets
    
    fprintf('Done.\n');
    
    if opts.do_val
        fprintf('Preparing validation data...');
        [image_roidb_val]...
                                = proposal_prepare_image_roidb(conf, opts.imdb_val, opts.roidb_val, bbox_means, bbox_stds);
        fprintf('Done.\n');

        % fix validation data
        shuffled_inds_val   = generate_random_minibatch([], image_roidb_val, conf.ims_per_batch);       % 1x4952 cell, 验证集所包含的images数量：4952x1(每个cell中包含一张图片索引，做为一个batch)
        shuffled_inds_val   = shuffled_inds_val(randperm(length(shuffled_inds_val), opts.val_iters));   % 1x500  cell, 只取500x1张图片做为做为验证集
    end
    
    conf.classes        = opts.imdb_train{1}.classes;
    
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  
    check_gpu_memory(conf, caffe_solver, opts.do_val);
     
%% -------------------- Training -------------------- 

    proposal_generate_minibatch_fun = @proposal_generate_minibatch;
    visual_debug_fun                = @proposal_visual_debug;

    % training
    shuffled_inds = [];
    train_results = [];  
    val_results = [];
    iter_ = caffe_solver.iter();
    max_iter = caffe_solver.max_iter();
    
    while (iter_ < max_iter)
        caffe_solver.net.set_phase('train');

        % generate minibatch training data
        % 第一次调用generate_random_minibatch()函数时，shuffled_inds = [], 根据传入的image_roidb_train(准备好的imdb+rois结合的结构体) conf.ims_per_batch(每次训练使用图片数:1张)，
        % 产生随机的训练数据索引，维度为1x4952 double。其中每一列包含两个一张图片索引，做为一个batch的训练图片。此后每次调用，只需要将shuffled_inds的第一列取出赋值给sub_inds做为训练的一张图片即可，
        % 然后删除shuffled_inds取出的第一列索引，下次直接再取第一列索引即可。
        [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, conf.ims_per_batch);
        % 计算每个batch所需要的blob格式的输入数据
        % input_blobs = {im_blob,           labels_blob,      label_weights_blob,    bbox_targets_blob,    bbox_loss_blob};
        % 1x5 cell array
        %           [800×600×3 single]    [51×39×9 single]    [51×39×9 single]       [51×39×36 single]    [51×39×36 single]
        [net_inputs, scale_inds] = proposal_generate_minibatch_fun(conf, image_roidb_train(sub_db_inds));
        
        
        % visual_debug_fun(conf, image_roidb_train(sub_db_inds), net_inputs, bbox_means, bbox_stds, conf.classes, scale_inds);
        caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);
        rst = caffe_solver.net.get_output();
        rst = check_error(rst, caffe_solver);
        train_results = parse_rst(train_results, rst);
        % check_loss(rst, caffe_solver, net_inputs);

        % do valdiation per val_interval iterations
        if ~mod(iter_, opts.val_interval) 
            if opts.do_val
                val_results = do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val);
            end
            
            show_state(iter_, train_results, val_results);
            train_results = [];
            diary; diary; % flush diary
        end
        
        % snapshot
        if ~mod(iter_, opts.snapshot_interval)
            snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
        end
        
        iter_ = caffe_solver.iter();
    end
    
    % final validation
    if opts.do_val
        do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val);
    end
    % final snapshot
    snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    save_model_path = snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
 
end

function val_results = do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val)
    val_results = [];

    caffe_solver.net.set_phase('test');
    for i = 1:length(shuffled_inds_val)
        sub_db_inds = shuffled_inds_val{i};
        [net_inputs, ~] = proposal_generate_minibatch_fun(conf, image_roidb_val(sub_db_inds));

        % Reshape net's input blobs
        caffe_solver.net.reshape_as_input(net_inputs);

        caffe_solver.net.forward(net_inputs);
        rst = caffe_solver.net.get_output();
        rst = check_error(rst, caffe_solver);  
        val_results = parse_rst(val_results, rst);
    end
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, ims_per_batch)
    % Input:
    %       image_roidb_train: 4952x1 struct
    %       ims_per_batch:     1
    % Function:
    %       shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory
        
        hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true); % 4952x1 logical
        vert_image_inds = ~hori_image_inds;
        hori_image_inds = find(hori_image_inds);  % 4065x1 logical
        vert_image_inds = find(vert_image_inds);  % 887x1  double
        
        % random perm
        lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;       % lim = 4065
        hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));  % 4065x1 double
        lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;       % lim = 886
        vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));  % 887x1  double
        
        % combine sample for each ims_per_batch 
        hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);              % 1x4065 double
        vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);              % 1x887  double
        
        shuffled_inds = [hori_image_inds, vert_image_inds];                         % 1x4952 double
        shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
        
        shuffled_inds = num2cell(shuffled_inds, 1); % 1x4952 cell
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function rst = check_error(rst, caffe_solver)

    cls_score = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data(); % 51x351x2 single cls_score为背景和前景的得分，其中背景得分在第一维：51x315x1, 前景得分在第二维。
    labels = caffe_solver.net.blobs('labels_reshape').get_data();                % 51x351x1 single   
    labels_weights = caffe_solver.net.blobs('labels_weights_reshape').get_data();% 51x351x1 single   
    
    accurate_fg = (cls_score(:, :, 2) > cls_score(:, :, 1)) & (labels == 1);     % 标签为1的，前景得分大于背景得分的做为前景分类正确的accurate_fg
    accurate_bg = (cls_score(:, :, 2) <= cls_score(:, :, 1)) & (labels == 0);    % 标签为0的，前景得分小于等于背景得分做为背景分类正确的accurate_bg
    accurate = accurate_fg | accurate_bg;                                        % 总的accurate 为二者取与 | 
    accuracy_fg = sum(accurate_fg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 1)) + eps); % 在所有前景中，前景分类正确的比例
    accuracy_bg = sum(accurate_bg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 0)) + eps); % 在所有背景中，背景分类正确的比例
    
    rst(end+1) = struct('blob_name', 'accuracy_fg', 'data', accuracy_fg);
    rst(end+1) = struct('blob_name', 'accuracy_bg', 'data', accuracy_bg);
end

function check_gpu_memory(conf, caffe_solver, do_val)
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough  

    % generate pseudo training data with max size
    im_blob = single(zeros(max(conf.scales), conf.max_size, 3, conf.ims_per_batch));
    
    anchor_num = size(conf.anchors, 1);
    output_width = conf.output_width_map.values({size(im_blob, 1)});
    output_width = output_width{1};
    output_height = conf.output_width_map.values({size(im_blob, 2)});
    output_height = output_height{1};
    labels_blob = single(zeros(output_width, output_height, anchor_num, conf.ims_per_batch));
    labels_weights = labels_blob;
    bbox_targets_blob = single(zeros(output_width, output_height, anchor_num*4, conf.ims_per_batch));
    bbox_loss_weights_blob = bbox_targets_blob;

    net_inputs = {im_blob, labels_blob, labels_weights, bbox_targets_blob, bbox_loss_weights_blob};
    
     % Reshape net's input blobs
    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);

    if do_val
        % use the same net with train to save memory
        caffe_solver.net.set_phase('test');
        caffe_solver.net.forward(net_inputs);
        caffe_solver.net.set_phase('train');
    end
end

function model_path = snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)
    anchor_size = size(conf.anchors, 1);
    bbox_stds_flatten = repmat(reshape(bbox_stds', [], 1), anchor_size, 1);
    bbox_means_flatten = repmat(reshape(bbox_means', [], 1), anchor_size, 1);
    
    % merge bbox_means, bbox_stds into the model
    bbox_pred_layer_name = 'proposal_bbox_pred';
    weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
    biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
    weights_back = weights;
    biase_back = biase;
    
    weights = ...
        bsxfun(@times, weights, permute(bbox_stds_flatten, [2, 3, 4, 1])); % weights = weights * stds; ? 为什么这里没有加上均值呢
    biase = ...
        biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;
    
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);
    
    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);
    
    % restore net to original state
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
end

function show_state(iter, train_results, val_results)
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    fprintf('Training : err_fg %.3g, err_bg %.3g, loss (cls %.3g + reg %.3g)\n', ...
        1 - mean(train_results.accuracy_fg.data), 1 - mean(train_results.accuracy_bg.data), ...
        mean(train_results.loss_cls.data), ...
        mean(train_results.loss_bbox.data));
    if exist('val_results', 'var') && ~isempty(val_results)
        fprintf('Testing  : err_fg %.3g, err_bg %.3g, loss (cls %.3g + reg %.3g)\n', ...
            1 - mean(val_results.accuracy_fg.data), 1 - mean(val_results.accuracy_bg.data), ...
            mean(val_results.loss_cls.data), ...
            mean(val_results.loss_bbox.data));
    end
end

function check_loss(rst, caffe_solver, input_blobs)
    im_blob = input_blobs{1};               % 800x600x3 single
    labels_blob = input_blobs{2};           % 51x39x9   single
    label_weights_blob = input_blobs{3};    % 51x39x9   single
    bbox_targets_blob = input_blobs{4};     % 51x39x36  single
    bbox_loss_weights_blob = input_blobs{5};% 51x39x36  single
    
    regression_output = caffe_solver.net.blobs('proposal_bbox_pred').get_data(); % 51x39x36 single
    % smooth l1 loss : 就是使用了论文中的SmoothL1loss公式，自己看一下就可以理解
    regression_delta = abs(regression_output(:) - bbox_targets_blob(:));          % 71604x1 single
    regression_delta_l2 = regression_delta < 1;                                   % 71604x1 logical,绝对值大于1的标记为1，其余的标记为0 
    regression_delta = 0.5 * regression_delta .* regression_delta .* regression_delta_l2 + (regression_delta - 0.5) .* ~regression_delta_l2;  % 71604x1 single
    regression_loss = sum(regression_delta.* bbox_loss_weights_blob(:)) / size(regression_output, 1) / size(regression_output, 2);  % 该张图片共有size(regression_output, 1) * size(regression_output, 2)，这么多个anchors，所以要将regression_loss除以anchors数目来做为这张图片bunuding boxes regression loss
    
    confidence = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data(); % 51x351x2 single
    labels = reshape(labels_blob, size(labels_blob, 1), []);                      % 51x351   single
    label_weights = reshape(label_weights_blob, size(label_weights_blob, 1), []); % 51x351   single
    
    % 分类层的loss计算要好好理解：
    % 首先使用softmax函数计算51x351个anchors属于背景和前景的概率得分，然后再reshape一下，得到17901x2维度。
    % 然后使用sub2ind将多维下标转化为1维索引，从而达到可以直接用这个索引取到每个anchors的值（labels为0，我们就会取到第一维的值；labels为1就会取到第二维的值）。
    % 就比如这里的sub2ind(size(confidence_softmax), 1:size(confidence_softmax, 1), labels(:)' + 1)
    % 同时转换多个元素的下标，共转换17901个
    % 横轴  1:17901这么多个数，代表该元素的所在横坐标。
    % 纵轴  labels(:)' + 1，代表该元素所在纵坐标。注意labels中只有0和1，我们这里通过加1，则将labels值都变为1和2了，就可以直接做为纵坐标来取了，
    %      labels+1为1的时候代表该anchors属于原始标签为0的背景，为2的时候表示属于原始标签为1前景。这样，我们就可以将这些anchors真实属于的类别confidence_loss都提取出来。
    confidence_softmax = bsxfun(@rdivide, exp(confidence), sum(exp(confidence), 3));     % 51x351x2 single
    confidence_softmax = reshape(confidence_softmax, [], 2);                             % 17901x2  single
    confidence_loss = confidence_softmax(sub2ind(size(confidence_softmax), 1:size(confidence_softmax, 1), labels(:)' + 1)); % 1x17901 single
    confidence_loss = -log(confidence_loss); % 1x17901 single
    confidence_loss = sum(confidence_loss' .* label_weights(:)) / sum(label_weights(:)); % 1x1 single
    
    results = parse_rst([], rst); % 验证使用CNN网络，即内部通过C++实现loss_cls 和 loss_bbox层的loss计算，是否和自己使用MATLAB计算的一致
    fprintf('C++   : conf %f, reg %f\n', results.loss_cls.data, results.loss_bbox.data);
    fprintf('Matlab: conf %f, reg %f\n', confidence_loss, regression_loss);
end