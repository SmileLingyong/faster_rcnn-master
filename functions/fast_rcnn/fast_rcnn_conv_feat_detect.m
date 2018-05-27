function [pred_boxes, scores] = fast_rcnn_conv_feat_detect(conf, caffe_net, im, conv_feat_blob, boxes, max_rois_num_in_gpu)
% [pred_boxes, scores] = fast_rcnn_conv_feat_detect(conf, caffe_net, im, conv_feat_blob, boxes, max_rois_num_in_gpu)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
%    该函数功能：将RPN网络conv5_3输出的feature map做为Fast-RCNN网络的输入'data'，
% 还有一个输入就是rois，该rois是RPN网络产生的，并且是对应于缩放后的原图上的.
% 因为可能会有GPU不足的情况，我们这里设置的max_rois_num_in_gpu=300，如果rois数量大于这个值，则需要将rois分批次出入，
% 每次最多输入300个rois到CNN中，并将输出的结果保存在该批次的cell数组中，最后将所有的cell数组结果使用cell2mat整合，
% 就得到该张图片conv5_3特征+rois，使用fast-rcnn（detection_test.prototxt网络）进行softmax分类+bounding boxes回归结果.
% 具体要查看这两个网络结构就可以比较容易明白:
%   ./output/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/proposal_test.prototxt
%   ./output/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/detection_test.prototxt
% ---------------------------------------------------------------

    [rois_blob, ~] = get_blobs(conf, im, boxes); % 维度：[feat_rois, levels]
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);
    rois_blob = single(rois_blob); % % 1x1x5x115
    
    % set conv feature map as 'data', 把RPN网络conv5_3产生的feature map做为data输入到 Fast-RCNN网络中
    caffe_net.blobs('data').copy_data_from(conv_feat_blob);
    
    total_rois = size(rois_blob, 4); % 115
    total_scores = cell(ceil(total_rois / max_rois_num_in_gpu), 1);     % 1x1 cell {[]}: because of in this case, ceil(115/300) = 1
    total_box_deltas = cell(ceil(total_rois / max_rois_num_in_gpu), 1); % 1x1 cell {[]}
    for i = 1:ceil(total_rois / max_rois_num_in_gpu)
        
        sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
        sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
        sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end); % 1x1x5x115
        
        % only set rois blob here
        net_inputs = {[], sub_rois_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        output_blobs = caffe_net.forward(net_inputs); % [84x115];[21x115]

        if conf.test_binary
            % simulate binary logistic regression
            scores = caffe_net.blobs('cls_score').get_data();
            scores = squeeze(scores)';
            % Return scores as fg - bg
            scores = bsxfun(@minus, scores, scores(:, 1));
        else
            % use softmax estimated probabilities
            scores = output_blobs{2};   % 21x115 single
            scores = squeeze(scores)';  % 115x21 single
        end

        % Apply bounding-box regression deltas
        box_deltas = output_blobs{1};       % 84x115 single
        box_deltas = squeeze(box_deltas)';  % 115x84 single
        
        total_scores{i} = scores;           % cell [115x21 single]  注意这里的技巧，先每个batch rois结果都放到cell中，最后再使用cell2mat整合，以满足合适GPU的要求
        total_box_deltas{i} = box_deltas;   % cell [115x84 single]
    end 
    
    scores = cell2mat(total_scores);  % 115X21 single
    box_deltas = cell2mat(total_box_deltas); % 115x84 single
    
    pred_boxes = fast_rcnn_bbox_transform_inv(boxes, box_deltas); % using bounding-box regression improve localization performance.
    pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));% 115x84 single
    
    % remove scores and boxes for back-ground
    pred_boxes = pred_boxes(:, 5:end);  % 115x80 single
    scores = scores(:, 2:end);          % 115x20 single
end

function [rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    im_scale_factors = get_image_blob_scales(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors); % 维度：[feat_rois, levels]
end

function im_scales = get_image_blob_scales(conf, im)
    im_scales = arrayfun(@(x) prep_im_for_blob_size(size(im), x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales); 
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
        levels = max(abs(scaled_areas - 224.^2), 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1; % 这里将rois按照scales比例缩放，即对应与使用缩放后的原图(短边=600)输入到CNN得到的feature map后，应该对应产生的rois(anchors boxes)尺寸
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
    