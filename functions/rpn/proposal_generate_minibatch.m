function [input_blobs, random_scale_inds] = proposal_generate_minibatch(conf, image_roidb)
% [input_blobs, random_scale_inds] = proposal_generate_minibatch(conf, image_roidb)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
% 函数功能：传入一个批次训练图片1张，每个训练batch使用256个rois(anchors), 
% 然后这256个rois中前景占50%,即128张，背景也占128张。然后计算该一个batch所需要的blob格式输入数据

    num_images = length(image_roidb);
    assert(num_images == 1, 'proposal_generate_minibatch_fcn only support num_images == 1');

    % Sample random scales to use for each image in this batch
    random_scale_inds = randi(length(conf.scales), num_images, 1);

    assert(mod(conf.batch_size, num_images) == 0, ...
        sprintf('num_images %d must divide BATCH_SIZE %d', num_images, conf.batch_size));
    
    rois_per_image = conf.batch_size / num_images;
    fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
    
    % Get the input image blob
    [im_blob, im_scales] = get_image_blob(conf, image_roidb, random_scale_inds);
    
    for i = 1:num_images
        [labels, label_weights, bbox_targets, bbox_loss] = ...
            sample_rois(conf, image_roidb(i), fg_rois_per_image, rois_per_image, im_scales(i), random_scale_inds(i));
        
        % get fcn output size
        img_size = round(image_roidb(i).im_size * im_scales(i));
        output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
        
        assert(img_size(1) == size(im_blob, 1) && img_size(2) == size(im_blob, 2));
        
        labels_blob = reshape(labels, size(conf.anchors, 1), output_size(1), output_size(2));   % 9x39x51,将一维的labels与使用CNN提取图片的conv5特征的维度大小进行对应，维度顺序还没有调整,目前是[channel, height, width] 
        label_weights_blob = reshape(label_weights, size(conf.anchors, 1), output_size(1), output_size(2));
        bbox_targets_blob = reshape(bbox_targets', size(conf.anchors, 1)*4, output_size(1), output_size(2));
        bbox_loss_blob = reshape(bbox_loss', size(conf.anchors, 1)*4, output_size(1), output_size(2));
        
        % permute from [channel, height, width], where channel is the
        % fastest dimension to [width, height, channel] 这样就调整为在CNN中的维度顺序，与图片在CNN中的维度顺序是对应的
        labels_blob = permute(labels_blob, [3, 2, 1]);  % 51x39x9
        label_weights_blob = permute(label_weights_blob, [3, 2, 1]);
        bbox_targets_blob = permute(bbox_targets_blob, [3, 2, 1]);
        bbox_loss_blob = permute(bbox_loss_blob, [3, 2, 1]);
    end
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));               % [800×600×3 single]
    labels_blob = single(labels_blob);
    labels_blob(labels_blob > 0) = 1; %to binary lable (fg and bg)  % [51×39×9 single] 注意这里细节，这里将labels标签大于1的都设置为1了，表示属于前景，而不需要知道前景到底是哪一类   
    label_weights_blob = single(label_weights_blob);                % [51×39×9 single]              
    bbox_targets_blob = single(bbox_targets_blob);                  % [51×39×36 single]
    bbox_loss_blob = single(bbox_loss_blob);                        % [51×39×36 single]
    
    assert(~isempty(im_blob));
    assert(~isempty(labels_blob));
    assert(~isempty(label_weights_blob));
    assert(~isempty(bbox_targets_blob));
    assert(~isempty(bbox_loss_blob));
    
    input_blobs = {im_blob, labels_blob, label_weights_blob, bbox_targets_blob, bbox_loss_blob};
    % input_blobs = {im_blob,           labels_blob,      label_weights_blob,    bbox_targets_blob,    bbox_loss_blob};
    % 1x5 cell array
    %           [800×600×3 single]    [51×39×9 single]    [51×39×9 single]       [51×39×36 single]    [51×39×36 single]
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
function [labels, label_weights, bbox_targets, bbox_loss_weights] = ...
    sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image, im_scale, im_scale_ind)
% 输入：
% 	image_roidb：		  一张图片的image_roidb结构体数据
% 	fg_rois_per_image = 128 : 256个rois中前景占50%即128张
% 	rois_per_image    = 256 : 一张图片使用256个rois (anchors)
% 输出：
% 	找到bbox_targets中前景rois，和背景rois，计算最终留下的前景和背景是哪些。
%   labels中，前景对应的设置为其ground truth的labels，背景的为0；
%   label_weights中，前景对应的为1，背景对应的为conf.bg_weight=1；
%   bbox_targets只留下后四维的平移缩放参数；
%   bbox_loss_weights中，前景对应的为1，背景为0

    bbox_targets = image_roidb.bbox_targets{im_scale_ind};
    ex_asign_labels = bbox_targets(:, 1);
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(bbox_targets(:, 1) > 0);
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(bbox_targets(:, 1) < 0);
    
    % select foreground
    fg_num = min(fg_rois_per_image, length(fg_inds));
    fg_inds = fg_inds(randperm(length(fg_inds), fg_num));
    
    bg_num = min(rois_per_image - fg_num, length(bg_inds));
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num));

    labels = zeros(size(bbox_targets, 1), 1);
    % set foreground labels
    labels(fg_inds) = ex_asign_labels(fg_inds);
    assert(all(ex_asign_labels(fg_inds) > 0));
    
    label_weights = zeros(size(bbox_targets, 1), 1);
    % set foreground labels weights
    label_weights(fg_inds) = 1;
    % set background labels weights
    label_weights(bg_inds) = conf.bg_weight;
    
    bbox_targets = single(full(bbox_targets(:, 2:end)));
    
    bbox_loss_weights = bbox_targets * 0;
    bbox_loss_weights(fg_inds, :) = 1;
end

function visual_anchors(image_roidb, anchors, im_scale)
    imshow(imresize(imread(image_roidb.image_path), im_scale));
    hold on;
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r'), num2cell(anchors, 2));
    hold off;
end

