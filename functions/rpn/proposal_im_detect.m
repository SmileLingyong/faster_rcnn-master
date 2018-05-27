function [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, caffe_net, im)
% [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, im, net_idx)
% 
%    函数功能：要结合proposal_test.prototxt网络以及proposal_locate_anchors()函数一起看。
% 其使用RPN网络，得到了RPN网络之后的proposal_bbox_pred 和 proposal_cls_prb（即可知道经过卷积之后的feature map大小），
% 然后计算该RPN网络最后输出的feature map中每个像素点，对应于原图中的位置。并使用该原图上的位置点，计算其在原图上的9个anchor(共有feature_map_size个位置)，
% 即可得到9×feature_map_size个anchor. 然后再用box_deltas(每个anchors需要做的平移尺度变换)，对产生的anchors进行边框回归位置精修。这样我们就得到了该张图片所需要的region proposals(anchors)
% 由output_blobs{2}:proposal_cls_prb[50x342x2]维，其表示经过分类层proposal_cls_score以及reshape——>softmax之后，输出的每个位置(共有50x38个位置)上, 9个anchor属于前景和背景的概率。
% 我们取第二维属于前景的概率得分，即得到了该张图片所有region proposals(anchors)属于前景的概率得分。
% [ 这样我们就得到了该张图片所需要的 pred_boxes + scores ]
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    im = single(im);
    [im_blob, im_scales] = get_image_blob(conf, im); % 缩放到短边为600的尺寸图片,以及缩放比例 600x800x3 1.6000
    im_size = size(im); % 375x500x3
    scaled_im_size = round(im_size * im_scales); % 600x800x5
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg  600x800x3
    im_blob = permute(im_blob, [2, 1, 3, 4]); % 800x600x3
    im_blob = single(im_blob); % 800x600x3

    net_inputs = {im_blob};

    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    output_blobs = caffe_net.forward(net_inputs); % 2cell [50x38x36];[50x342x2]

    % Apply bounding-box regression deltas
    box_deltas = output_blobs{1}; % 50x38x36: 对应于[width, height, channel]
    featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)]; % 38x50, 对应于[hight, width]注意这里的细节，这里先取了height，然后再取的width
    % permute from [width, height, channel] to [channel, height, width], where channel is the fastest dimension
    % 这样box_deltas就和featuremap对应到了，都是[height, width]这样的优先顺序,对应于缩放后的原图尺寸就是600x800
    box_deltas = permute(box_deltas, [3, 2, 1]); % 36x38x50, 
    box_deltas = reshape(box_deltas, 4, [])';    % 17100x4
    
    anchors = proposal_locate_anchors(conf, size(im), conf.test_scales, featuremap_size); %% #lly# see function detail, 此时得到的是缩放后的原图尺寸对应的anchors,(短边为600的)
    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas); %% #lly# see function detail, 用box_deltas(每个anchors需要做的平移尺度变换)，对产生的anchors进行边框回归位置精修。
      % scale back : 将anchors缩放回去，即缩放到原图尺寸 375x500 对应的anchors
    pred_boxes = bsxfun(@times, pred_boxes - 1, ...
        ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
    pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1)); %% #lly# see function detail,相当于规范anchors的边界，把anchors超过原图375x500边界的部分都减去
    
    assert(conf.test_binary == false);
    % use softmax estimated probabilities
    scores = output_blobs{2}(:, :, end); % 50x342, 说明: output_blobs{2}:[51x351x2]维，其表示经过分类层proposal_cls_score以及reshape——>softmax之后，输出的每个位置(共有51x39个位置)上, 9个anchor属于前景和背景的概率。我们这里只取后面属于前景的概率得分
    scores = reshape(scores, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []); % 50x38x9
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    scores = permute(scores, [3, 2, 1]); % 9x38x50 single
    scores = scores(:); % 17100x1 single
    
    box_deltas_ = box_deltas;   % 17100x4 single
    anchors_ = anchors;         % 17100x4 double
    scores_ = scores;           % 17100x1 single
    
    if conf.test_drop_boxes_runoff_image
        contained_in_image = is_contain_in_image(anchors, round(size(im) * im_scales));
        pred_boxes = pred_boxes(contained_in_image, :);
        scores = scores(contained_in_image, :);
    end
    
    % drop too small boxes
    [pred_boxes, scores] = filter_boxes(conf.test_min_box_size, pred_boxes, scores);
    
    % sort
    [scores, scores_ind] = sort(scores, 'descend');
    pred_boxes = pred_boxes(scores_ind, :);
end

function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

%% [blob, im_scales] = get_image_blob(conf, im)
% get blob which is a scaled image, and get image scales.
function [blob, im_scales] = get_image_blob(conf, im)
    if length(conf.test_scales) == 1
        [blob, im_scales] = prep_im_for_blob(im, conf.image_means, conf.test_scales, conf.test_max_size);
    else
        [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
        im_scales = cell2mat(im_scales);
        blob = im_list_to_blob(ims);    
    end
end

%% drop too small boxes
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
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels)) / conf.feat_stride) + 1;
end

function [boxes, scores] = filter_boxes(min_box_size, boxes, scores)
    widths = boxes(:, 3) - boxes(:, 1) + 1;
    heights = boxes(:, 4) - boxes(:, 2) + 1;
    
    valid_ind = widths >= min_box_size & heights >= min_box_size;
    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
end
    
function boxes = clip_boxes(boxes, im_width, im_height)
    % #lly# 保证anchor都在图片内，即将x1规范到1～im_width范围，x1<1 会被置为1，x1>im_width会被置为im_width. 其余的类似操作
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end

function contained = is_contain_in_image(boxes, im_size)
    contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
    
    contained = all(contained, 2);
end
    
