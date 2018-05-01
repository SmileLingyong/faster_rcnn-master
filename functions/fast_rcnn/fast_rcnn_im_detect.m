function [pred_boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% [pred_boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    [im_blob, rois_blob, ~] = get_blobs(conf, im, boxes); % 将images和rois(boxes) 按同样尺寸缩放(短边=600)，做为输入到网络中的blob
    
    % When mapping from image ROIs to feature map ROIs, there's some aliasing
    % (some distinct image ROIs get mapped to the same feature ROI).
    % Here, we identify duplicate feature ROIs, so we only compute features
    % on the unique subset.
    [~, index, inv_index] = unique(rois_blob, 'rows');
    rois_blob = rois_blob(index, :);  % 2608x5
    boxes = boxes(index, :);          % 2608c4
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg    % 896x600x3
    im_blob = permute(im_blob, [2, 1, 3, 4]);                   % 600x896x3
    im_blob = single(im_blob);
    rois_blob = rois_blob - 1; % to c's index (start from 0)    % 2608x5
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);               % 1x1x5x2608
    rois_blob = single(rois_blob);                              
    
    total_rois = size(rois_blob, 4);  % 2608
    total_scores = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    total_box_deltas = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    for i = 1:ceil(total_rois / max_rois_num_in_gpu)  % When rois is excessive, due to GPU restrictions, batch training
        
        sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
        sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
        sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end); 
        
        net_inputs = {im_blob, sub_rois_blob}; % [600x896x3 single];[1x1x5x500 single]

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        output_blobs = caffe_net.forward(net_inputs); % [84x500 single]; [21x500 single]

        if conf.test_binary
            % simulate binary logistic regression
            scores = caffe_net.blobs('cls_score').get_data();
            scores = squeeze(scores)';
            % Return scores as fg - bg
            scores = bsxfun(@minus, scores, scores(:, 1));
        else
            % use softmax estimated probabilities
            scores = output_blobs{2};   % 21x500 single
            scores = squeeze(scores)';  % 500x21 single
        end

        % Apply bounding-box regression deltas
        box_deltas = output_blobs{1};       % 84x500
        box_deltas = squeeze(box_deltas)';  % 500x84
        
        total_scores{i} = scores;
        total_box_deltas{i} = box_deltas;
    end 
    
    scores = cell2mat(total_scores);         % 2608x21
    box_deltas = cell2mat(total_box_deltas); % 2608x84
    
    pred_boxes = fast_rcnn_bbox_transform_inv(boxes, box_deltas); % 2608x84 using bounding-box regression improve localization performance.
    pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));

    % Map scores and predictions back to the original set of boxes
    scores = scores(inv_index, :);           % 2608x21
    pred_boxes = pred_boxes(inv_index, :);   % 2608x84
    
    % remove scores and boxes for back-ground
    pred_boxes = pred_boxes(:, 5:end);       % 2608x80
    scores = scores(:, 2:end);               % 2608x20
end

function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function [blob, im_scales] = get_image_blob(conf, im)
    [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales);
    blob = im_list_to_blob(ims);    
end

function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
    [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
    rois_blob = single([levels, feat_rois]);
end

function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)
    im_rois = single(im_rois);
    
    if length(scales) > 1
        widths = im_rois(:, 3) - im_rois(:, 1) + 1;
        heights = im_rois(:, 4) - im_rois(:, 2) + 1;
        
        areas = widths .* heights;
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
        [~, levels] = min(abs(scaled_areas - 224.^2), [], 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1; % 这里将rois(原图上的region proposals)按照scales比例(原图缩放比例)缩放。这样才能使缩放后的原图输入CNN得到feature map，能够找到rois如果输入到CNN应该对应的feature map
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % #lly# 保证boxes都在图片内，即将x1规范到1～im_width范围，x1<1 会被置为1，x1>im_width会被置为im_width. 其余的类似操作
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end
    