function [image_roidb, bbox_means, bbox_stds] = proposal_prepare_image_roidb(conf, imdbs, roidbs, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = proposal_prepare_image_roidb(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
% 函数功能：传入image_roidb结构体数据，然后将数据的每张图片使用proposal_locate_anchors()函数计算其对应产生的anchors，以及图片缩放的尺度。
% 接着使用compute_targets()函数，计算该张图片中的anchors哪些做为前景fg，哪些做为背景bg，
% 然后计算该张图片中这些前景回归到其对应的ground truth所需要做的平移尺度变换参数，加上其所属类别；
% 背景则没有平移尺度变换参数，4个参数为0，加上其所属类别为-1；所以每个bbox_targets都是[labels(ex_inds), regression_label] 5维的数据
% 然后计算所有图片的所有boxes_targets的均值和方差.并将bboxes_targets使用该均值和方差进行归一化处理(具体细节如下)
% ---------------------------------------------------------------------------------------------------------------------

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
    
    % 将传入的imdbs和roidbs结构体数据构造一个新的结构体数据，这里我们只时使用每个roidbs的ground truth做为boxes
    if conf.target_only_gt
        image_roidb = ...
            cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                    struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, 'num_classes', x.num_classes, ...
                    'boxes', y.rois(z).boxes(y.rois(z).gt, :), 'class', y.rois(z).class(y.rois(z).gt, :), 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
            imdbs, roidbs, 'UniformOutput', false);
    else % 如果不是conf.target_only_gt，则使用每个roidbs所有的boxes
        image_roidb = ...
            cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                    struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                    'boxes', y.rois(z).boxes, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
            imdbs, roidbs, 'UniformOutput', false);
    end
   
    image_roidb = cat(1, image_roidb{:});
    
    % enhance roidb to contain bounding-box regression targets
    [image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end

% 计算所有图片的image_roidb(i).bbox_targets的均值bbox_means和方差bbox_stds，并将image_roidb(i).bbox_targets使用该均值和方差进行归一化处理.
function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
    % means and stds -- (k+1) * 4, include background class
% 传入image_roidb的结构体数据，使用proposal_locate_anchors()函数，计算每个image_roidb_cell{i}这一张图片对应产生的anchors，和图片的缩放尺度im_scales。
% 然后将每张图片的这些anchors，im_scales，images_roidb中的结构体数据，使用compute_targets()函数来计算该张图片中的哪些anchors属于前景，哪些anchors属于背景，
% 并找到每个前景对应的ground truth，然后计算将这些前景边框回归到ground truth所需要做的平移缩放参数regression_label，
% 加上每个前景anchors所属的ground truth类别标签gt_labels，[gt_labels; regression_label]共5维数据做为该张图片前景的bbox_targets；
% 属于背景的anchors不需要做边框回归，因此其regression_label四个参数对应为0，gt_labels为-1。这样就计算得到了每张图片的bbox_targets参数。 
% 然后计算所有图片的所有boxes_targets的均值和方差.并将bboxes_targets使用该均值和方差进行归一化处理
    
    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    image_roidb_cell = num2cell(image_roidb, 2);
    bbox_targets = cell(num_images, 1);
    parfor i = 1:num_images % 并行for循环
        % for fcn, anchors are concated as [channel, height, width], where channel is the fastest dimension.
       [anchors, im_scales] = proposal_locate_anchors(conf, image_roidb_cell{i}.im_size);
        
       gt_rois = image_roidb_cell{i}.boxes;
       gt_labels = image_roidb_cell{i}.class;
       im_size = image_roidb_cell{i}.im_size;
       bbox_targets{i} = cellfun(@(x, y) ...
           compute_targets(conf, scale_rois(gt_rois, im_size, y), gt_labels,  x, image_roidb_cell{i}, y), ...
           anchors, im_scales, 'UniformOutput', false);
    end
    clear image_roidb_cell;
    for i = 1:num_images
        image_roidb(i).bbox_targets = bbox_targets{i};
    end
    clear bbox_targets;
    
    % 计算所有图片的所有boxes_targets的均值和方差
    % 计算所有图片的中bbox_targets中前景的总个数，做为class_counts，
    % 然后将这些前景的bbox_targets的平移缩放参数4维度分别求和得到sums，
    % 4个维度平方后再求和得到squared_sums。
    % 然后将sums/class_counts得到所有boxes_targets的均值means.
    % 并使用squared_sums/class_counts - means.^2得到方差。(即平方的期望减去期望的平方)
    % 就得到最终的，所有图片所有boxes_targets的均值和方差。
    if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
        % Compute values needed for means and stds
        % var(x) = E(x^2) - E(x)^2
        class_counts = zeros(1, 1) + eps;
        sums = zeros(1, 4);
        squared_sums = zeros(1, 4);
        for i = 1:num_images
           for j = 1:length(conf.scales)
                targets = image_roidb(i).bbox_targets{j};
                gt_inds = find(targets(:, 1) > 0);
                if ~isempty(gt_inds)
                    class_counts = class_counts + length(gt_inds); 
                    sums = sums + sum(targets(gt_inds, 2:end), 1);
                    squared_sums = squared_sums + sum(targets(gt_inds, 2:end).^2, 1);
                end
           end
        end

        means = bsxfun(@rdivide, sums, class_counts);
        stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
    end
    
    % Normalize targets
    for i = 1:num_images
        for j = 1:length(conf.scales)
            targets = image_roidb(i).bbox_targets{j};
            gt_inds = find(targets(:, 1) > 0);
            if ~isempty(gt_inds)
                image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), means);
                image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), stds);
            end
        end
    end
end

function scaled_rois = scale_rois(rois, im_size, im_scale)
    im_size_scaled = round(im_size * im_scale);
    scale = (im_size_scaled - 1) ./ (im_size - 1);
    scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end

function bbox_targets = compute_targets(conf, gt_rois, gt_labels, ex_rois, image_roidb, im_scale)
% output: bbox_targets
%   positive: [class_label, regression_label]
%   ingore: [0, zero(regression_label)]
%   negative: [-1, zero(regression_label)]
% 
% 函数功能:
%    计算该张图片中的哪些anchors属于前景，哪些anchors属于背景，
% 并找到每个前景对应的ground truth，然后计算将这些前景边框回归到ground truth所需要做的平移缩放参数regression_label，
% 加上每个前景anchors所属的ground truth类别标签gt_labels，[gt_labels; regression_label]共5维数据做为该张图片前景的bbox_targets；
% 属于背景的anchors不需要做边框回归，因此其regression_label四个参数对应为0，gt_labels为-1。这样就计算得到了每张图片的bbox_targets参数。    
    if isempty(gt_rois)
        bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
        bbox_targets(:, 1) = -1;
        return;
    end

    % ensure gt_labels is in single
    gt_labels = single(gt_labels);
    assert(all(gt_labels > 0));

    % calc overlap between ex_rois(anchors) and gt_rois:计算该张图片每个anchors与每个ground truth的overlap
    ex_gt_overlaps = boxoverlap(ex_rois, gt_rois);
    
    % drop anchors which run out off image boundaries, if necessary
    if conf.drop_boxes_runoff_image
         contained_in_image = is_contain_in_image(ex_rois, round(image_roidb.im_size * im_scale));
         ex_gt_overlaps(~contained_in_image, :) = 0;
    end

    % for each ex_rois(anchors), get its max overlap with all gt_rois
    % ex_max_overlaps：ex_gt_overlaps每行最大值，每个anchors与哪个ground truth的IOU最大
    % ex_assignment： 每个anchors属于哪个anchors
    [ex_max_overlaps, ex_assignment] = max(ex_gt_overlaps, [], 2);
    
    % for each gt_rois, get its max overlap with all ex_rois(anchors), the
    % ex_rois(anchors) are recorded in gt_assignment
    % gt_assignment will be assigned as positive 
    % (assign a rois for each gt at least) 
    % gt_max_overlaps：ex_gt_overlaps每列最大值，即每个ground truth与所有anchors的IOU的最大值
    % gt_assignment： 这些IOU相交的最大值是那几个anchors交出来的。
    [gt_max_overlaps, gt_assignment] = max(ex_gt_overlaps, [], 1); 
    
    % ex_rois(anchors) with gt_max_overlaps maybe more than one, find them
    % as (gt_best_matches): 与这些ground truth相交得到最大值gt_max_overlaps的anchors可能有多个: gt_best_matches
    [gt_best_matches, gt_ind] = find(bsxfun(@eq, ex_gt_overlaps, [gt_max_overlaps]));
    
    % Indices of examples for which we try to make predictions
    % both (ex_max_overlaps >= conf.fg_thresh) and gt_best_matches are
    % assigned as positive examples: 找到(ex_max_overlaps >= conf.fg_thresh)即IOU >= conf.fg_thresh的，
    % 以及上面找到的gt_best_matches，去掉重复的，做为该张图片anchors中属于前景的部分
    fg_inds = unique([find(ex_max_overlaps >= conf.fg_thresh); gt_best_matches]);
        
    % Indices of examples for which we try to used as negtive samples
    % the logic for assigning labels to anchors can be satisfied by both the positive label and the negative label
    % When this happens, the code gives the positive label precedence to
    % pursue high recall: 这些anchors中找到其属于背景的部分
    bg_inds = setdiff(find(ex_max_overlaps < conf.bg_thresh_hi & ex_max_overlaps >= conf.bg_thresh_lo), ...
                    fg_inds);
    
    if conf.drop_boxes_runoff_image
        contained_in_image_ind = find(contained_in_image);
        fg_inds = intersect(fg_inds, contained_in_image_ind);
        bg_inds = intersect(bg_inds, contained_in_image_ind);
    end
                
    % Find which gt ROI each ex ROI has max overlap with:
    % this will be the ex ROI's gt target: 将前景属于类别的ground truth做为target_rois
    target_rois = gt_rois(ex_assignment(fg_inds), :);
    src_rois = ex_rois(fg_inds, :); % 前景rois,即boxes
    
    % we predict regression_label which is generated by an un-linear
    % transformation from src_rois and target_rois: 
    [regression_label] = fast_rcnn_bbox_transform(src_rois, target_rois); % 将这些前景boxes与其所属的ground truth计算其做边框回归所需要做的平移缩放参数
    
    bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
    bbox_targets(fg_inds, :) = [gt_labels(ex_assignment(fg_inds)), regression_label]; % 前景的bbox_targets由其所属的gt_labels + regression_label组成
    bbox_targets(bg_inds, 1) = -1; % 背景的bbox_targets中gt_labels为-1，不需要做边框回归，所有regression_label中的4个参数都为0
    
    if 0 % debug
        %%%%%%%%%%%%%%
        im = imread(image_roidb.image_path);
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, conf.scales, conf.max_size);
        imshow(mat2gray(im));
        hold on;
        cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r'), ...
                   num2cell(src_rois, 2));
        cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'g'), ...
                   num2cell(target_rois, 2));
        hold off;
        %%%%%%%%%%%%%%
    end
    
    bbox_targets = sparse(bbox_targets);
end

function contained = is_contain_in_image(boxes, im_size)
    contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
    
    contained = all(contained, 2);
end