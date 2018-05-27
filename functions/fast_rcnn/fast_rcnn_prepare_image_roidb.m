function [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb(conf, imdbs, roidbs, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
%   Gather useful information from imdb and roidb
%   pre-calculate mean (bbox_means) and std (bbox_stds) of the regression
%   term for normalization
% 
% 输入：
%     imdbs(图片数据) + roidbs(候选框数据)
% 输出：
%     计算以下参数：其中包括，计算所有图片的image_roidb(i).bbox_targets的均值bbox_means和方差bbox_stds，并将image_roidb(i).bbox_targets使用该均值和方差进行归一化处理.
%     image_roidb   : 10022x1 struct array with : 该参数主要理解bbox_targets的含义。
%             image_path
%             image_id
%             im_size
%             imdb_name
%             overlap   : (每个[ground truth + region proposals] (boxes)与每个ground truth的重叠率[若一个boxes与同类的多个ground truth重叠率取最大的那个])
%             boxes     : [ground truth + region proposals] (boxes)
%             class     
%             image
%             bbox_targets  : 每个ex_rois与其gt_rois进行bounding boxes regression需要做的平移尺度变换(4个参数)。 再加上其所属ground truth类别标签(1参数)，构成bbox_targets
%     bbox_means        : 21x4 double, 所有训练图片的rois的bbox_means均值，bbox_stds方差。
%     bbox_stds         : 21x4 double
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------- 
    
    if ~exist('bbox_means', 'var')
        bbox_means = [];
        bbox_stds = [];
    end
    
    if ~iscell(imdbs)
        imdbs = {imdbs};
        roidbs = {roidbs};
    end

    imdbs = imdbs(:);
    roidbs = roidbs(:);
    
    image_roidb = ...
        cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                        struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                        'overlap', y.rois(z).overlap, 'boxes', y.rois(z).boxes, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
        imdbs, roidbs, 'UniformOutput', false);
    
    image_roidb = cat(1, image_roidb{:});
    
    % enhance roidb to contain bounding-box regression targets
    [image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end

% 计算所有图片的image_roidb(i).bbox_targets的均值bbox_means和方差bbox_stds，并将image_roidb(i).bbox_targets使用该均值和方差进行归一化处理.
function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
    % means and stds -- (k+1) * 4, include background class

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    valid_imgs = true(num_images, 1);
    for i = 1:num_images
       rois = image_roidb(i).boxes; 
       [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
           compute_targets(conf, rois, image_roidb(i).overlap); % #lly# see function detail. 计算每张图片rois的bbox_targets，即每张图片的rois(每个region proposals)属于哪个ground truth的类，以及需要做的平移尺度变换。并检测是否使用这张图片
    end
    if ~all(valid_imgs)
        image_roidb = image_roidb(valid_imgs);
        num_images = length(image_roidb);
        fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
    end
    
    % 计算所有图片的所有boxes_targets的均值和方差
    % 计算20类中，每一类所包含的boxes_targets(表示的是每个前景boxes回归到其对应的ground truth所需要做的平移尺度变换。前景才有值，背景都为0)进行求和sums，以及平方求和squared_sums。
    % 然后将sums/class_counts得到所有boxes_targets的均值means.
    % 并使用squared_sums/class_counts - means.^2得到方差。(即平方的期望减去期望的平方)
    % 就得到最终的，所有图片所有boxes_targets的均值和方差。
    if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds)) % 计算所有图片的rois的均值和方差
        % Compute values needed for means and stds
        % var(x) = E(x^2) - E(x)^2
        class_counts = zeros(num_classes, 1) + eps; % 20x1
        sums = zeros(num_classes, 4);               % 20x4
        squared_sums = zeros(num_classes, 4);       % 20x4
        for i = 1:num_images
           targets = image_roidb(i).bbox_targets;
           for cls = 1:num_classes
              cls_inds = find(targets(:, 1) == cls);
              if ~isempty(cls_inds)
                 class_counts(cls) = class_counts(cls) + length(cls_inds); 
                 sums(cls, :) = sums(cls, :) + sum(targets(cls_inds, 2:end), 1);
                 squared_sums(cls, :) = squared_sums(cls, :) + sum(targets(cls_inds, 2:end).^2, 1);
              end
           end
        end

        means = bsxfun(@rdivide, sums, class_counts);
        stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
        
        % add background class
        means = [0, 0, 0, 0; means]; 
        stds = [0, 0, 0, 0; stds];
    end
    
    % Normalize targets
    for i = 1:num_images
        targets = image_roidb(i).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb(i).bbox_targets(cls_inds, 2:end), means(cls+1, :));
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb(i).bbox_targets(cls_inds, 2:end), stds(cls+1, :));
            end
        end
    end
end

%% function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap)
% 传入每一张图片的rois(region proposals)，overlap(每个[ground truth + region proposals] (boxes)与每个ground truth的重叠率[若一个boxes与同类的多个ground truth重叠率取最大的那个])。
% 然后，找到overlap每行的最大值，以及该最大值属于哪一类，即可判断该rois的每个boxes属于哪一类。然后再筛选overlap大于conf.bbox_thresh的，即筛选出属于前景的boxes。 
% 然后将这些属于前景的boxes与每个ground truth计算IOU，取每个前景boxes计算的最大的IOU的ground truth做为其 ground truth。
% 然后计算每个ex_rois与其gt_rois进行bounding boxes regression需要做的平移尺度变换(4个参数)。 再加上其所属ground truth类别标签(1参数)，构成bbox_targets。
% Input:    
%       rois:       2246x4  single
%       overlap:    2246x20 single
% Output:
%       bbox_targets: 2246x5 single
%       is_valid:     1x1    logical
function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap)

    overlap = full(overlap);
    
    [max_overlaps, max_labels] = max(overlap, [], 2); % 找到overlap每行的最大值，以及该最大值属于哪一类，即可判断该rois的每个boxes属于哪一类

    % ensure ROIs are floats
    rois = single(rois);
    
    bbox_targets = zeros(size(rois, 1), 5, 'single');
    
    % Indices of ground-truth ROIs, 找到ground truth的索引
    gt_inds = find(max_overlaps == 1); 
    
    if ~isempty(gt_inds)
        % Indices of examples for which we try to make predictions
        ex_inds = find(max_overlaps >= conf.bbox_thresh);

        % Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = boxoverlap(rois(ex_inds, :), rois(gt_inds, :));

        assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));

        % Find which gt ROI each ex ROI has max overlap with:
        % this will be the ex ROI's gt target
        [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
        gt_rois = rois(gt_inds(gt_assignment), :);
        ex_rois = rois(ex_inds, :);

        [regression_label] = fast_rcnn_bbox_transform(ex_rois, gt_rois);

        bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
    end
    
    % Select foreground ROIs as those with >= fg_thresh overlap
    is_fg = max_overlaps >= conf.fg_thresh;
    % Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
    is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;
    
    % check if there is any fg or bg sample. If no, filter out this image,检查该张图片的rois(region proposals boxes)是否存在属于前景和背景的。如果都不属于也不属于背景，则舍弃这张图片不用
    is_valid = true;
    if ~any(is_fg | is_bg)
        is_valid = false;
    end
end