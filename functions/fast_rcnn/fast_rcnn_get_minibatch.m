function [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob] = fast_rcnn_get_minibatch(conf, image_roidb)
% [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob] ...
%    = fast_rcnn_get_minibatch(conf, image_roidb)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
% 函数功能：传入一个批次batch的训练图片2张的结构体参数，每个训练batch使用128个rois(region proposals)，
% 即每张图片使用64个rois，然后这64个rois中前景占25%即16张，背景占75%即48张。然后计算该一个batch所需要的blob格式输入数据

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    % Sample random scales to use for each image in this batch
    random_scale_inds = randi(length(conf.scales), num_images, 1);
    
    assert(mod(conf.batch_size, num_images) == 0, ...
        sprintf('num_images %d must divide BATCH_SIZE %d', num_images, conf.batch_size));
    
    rois_per_image = conf.batch_size / num_images;
    fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
    
    % Get the input image blob
    [im_blob, im_scales] = get_image_blob(conf, image_roidb, random_scale_inds);
    
    % build the region of interest and label blobs
    rois_blob = zeros(0, 5, 'single');
    labels_blob = zeros(0, 1, 'single');
    bbox_targets_blob = zeros(0, 4 * (num_classes+1), 'single');
    bbox_loss_blob = zeros(size(bbox_targets_blob), 'single');
    
    for i = 1:num_images
        [labels, ~, im_rois, bbox_targets, bbox_loss] = ...
            sample_rois(conf, image_roidb(i), fg_rois_per_image, rois_per_image);
        
        % Add to ROIs blob
        feat_rois = fast_rcnn_map_im_rois_to_feat_rois(conf, im_rois, im_scales(i)); % 将rois进行尺度缩放，该尺度是其对应图片get_image_blob所做的尺度缩放，这样就与得到的feature map能够进行映射
        batch_ind = i * ones(size(feat_rois, 1), 1);    % 训练batch的索引？不太理解这个索引的作用
        rois_blob_this_image = [batch_ind, feat_rois];  % [batch_ind, feat_rois]构成了该张图片的rois blob
        rois_blob = [rois_blob; rois_blob_this_image];  % 将该batch的所有图片(2张)的rois_blob组合，构成一个batch的rois_blob
        
        % Add to labels, bbox targets, and bbox loss blobs
        % 最后得到一个batch(2张图片，128个rois)的
        labels_blob = [labels_blob; labels];                    % 128x1
        bbox_targets_blob = [bbox_targets_blob; bbox_targets];  % 128x84
        bbox_loss_blob = [bbox_loss_blob; bbox_loss];           % 128x84
    end
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = single(permute(rois_blob, [3, 4, 2, 1]));
    labels_blob = single(permute(labels_blob, [3, 4, 2, 1]));
    bbox_targets_blob = single(permute(bbox_targets_blob, [3, 4, 2, 1])); 
    bbox_loss_blob = single(permute(bbox_loss_blob, [3, 4, 2, 1]));
    
    assert(~isempty(im_blob));
    assert(~isempty(rois_blob));
    assert(~isempty(labels_blob));
    assert(~isempty(bbox_targets_blob));
    assert(~isempty(bbox_loss_blob));
end

%% Build an input blob from the images in the roidb at the specified scales.
function [im_blob, im_scales] = get_image_blob(conf, images, random_scale_inds)
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    for i = 1:num_images
        im = imread(images(i).image_path);
        target_size = conf.scales(random_scale_inds(i));
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

%% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, overlaps, rois, bbox_targets, bbox_loss_weights] = ...
    sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image)
% 输入：
%       image_roidb :            一张图片的image_roidb结构体参数
%       fg_rois_per_image = 16 : 64个rois中前景占25%即16张
%       rois_per_image    = 64 : 一张图片使用64个rois(region proposal)
% 输出： 计算传入的该张图片所保留的64个rois(16个前景+48个背景rois)，
%       以及每个rois所对应的labels(背景为0)，每个rois的overlap，
%       以及每个rois对应的bbox_targets，和 bbox_loss_weights。
%       (其中bbox_targets只是将 64x5维的数据，存储于64x84维的矩阵中而已）(详细的bbox_targets由来，亲查看fast_rcnn_prepare_image_roidb.m中的compute_targets()函数)
    [overlaps, labels] = max(image_roidb(1).overlap, [], 2); % 计算每行overlap中最大的，以及其所属的列(即类别)
%     labels = image_roidb(1).max_classes;
%     overlaps = image_roidb(1).max_overlaps;
    rois = image_roidb(1).boxes;
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(overlaps >= conf.fg_thresh);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image = min(fg_rois_per_image, length(fg_inds));
    % Sample foreground regions without replacement
    if ~isempty(fg_inds)
       fg_inds = fg_inds(randperm(length(fg_inds), fg_rois_per_this_image));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(overlaps < conf.bg_thresh_hi & overlaps >= conf.bg_thresh_lo);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image;
    bg_rois_per_this_image = min(bg_rois_per_this_image, length(bg_inds));
    % Sample foreground regions without replacement
    if ~isempty(bg_inds)
       bg_inds = bg_inds(randperm(length(bg_inds), bg_rois_per_this_image));
    end
    % The indices that we're selecting (both fg and bg)
    keep_inds = [fg_inds; bg_inds];
    % Select sampled values from various arrays
    labels = labels(keep_inds);
    % Clamp labels for the background ROIs to 0
    labels((fg_rois_per_this_image+1):end) = 0;
    overlaps = overlaps(keep_inds);
    rois = rois(keep_inds, :);
    
    assert(all(labels == image_roidb.bbox_targets(keep_inds, 1)));
    
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    
    [bbox_targets, bbox_loss_weights] = get_bbox_regression_labels(conf, ...
        image_roidb.bbox_targets(keep_inds, :), num_classes);
    
end

function [bbox_targets, bbox_loss_weights] = get_bbox_regression_labels(conf, bbox_target_data, num_classes)
%% Bounding-box regression targets are stored in a compact form in the roidb.
 % This function expands those targets into the 4-of-4*(num_classes+1) representation used
 % by the network (i.e. only one class has non-zero targets).
 % The loss weights are similarly expanded.
% Input bbox_target_data: [64x5 single]
% Return (N, (num_classes+1) * 4, 1, 1) blob of regression targets
% Return (N, (num_classes+1) * 4, 1, 1) blob of loss weights.  
% bbox_target_data : 64x5,
% 即将bbox_target_data 64x5维的数据，存储于64x84维的bbox_targets矩阵中。
% 比如bbox_target_data中的前14个rois是属于第7类的前景，然后我就将这前14个rois的后4个bbox_target_data数据(平移尺度变换数据) 
% 存储于前14行的[1+7*4 : (7+1)*4]列,(即(29,30,31,32)列)。
% 对应的bbox_loss_weights的(29,30,31,32)列会存放1，表示前景
    clss = bbox_target_data(:, 1);                                      % clss: 64x1
    bbox_targets = zeros(length(clss), 4 * (num_classes+1), 'single');  % bbox_targets: 64x84
    bbox_loss_weights = zeros(size(bbox_targets), 'single');            % bbox_loss_weights: 64x84
    inds = find(clss > 0);
    for i = 1:length(inds)
       ind = inds(i);
       cls = clss(ind);
       targets_inds = (1+cls*4):((cls+1)*4);
       bbox_targets(ind, targets_inds) = bbox_target_data(ind, 2:end);
       bbox_loss_weights(ind, targets_inds) = 1;  
    end
end


