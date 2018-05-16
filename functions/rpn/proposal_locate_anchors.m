function [anchors, im_scales] = proposal_locate_anchors(conf, im_size, target_scale, feature_map_size)
% [anchors, im_scales] = proposal_locate_anchors(conf, im_size, target_scale, feature_map_size)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
% generate anchors for each scale
% 该函数功能主要是：计算使用proposal_test.prototxt中RPN网络输出的feature map中每个像素点，对应于原图(缩放后的)中的位置。
% 然后，使用映射回原图(缩放后的)上的点，计算其在原图(缩放后的)上的9个anchor(共有feature_map_size个位置)，即可得到9×feature_map_size个anchor
% 注意：这里是缩放后的原图，因为输入CNN网络时就是缩放后的原图

    % only for fcn
    if ~exist('feature_map_size', 'var')
        feature_map_size = [];
    end

    func = @proposal_locate_anchors_single_scale;

    if exist('target_scale', 'var')
        [anchors, im_scales] = func(im_size, conf, target_scale, feature_map_size);
    else
        [anchors, im_scales] = arrayfun(@(x) func(im_size, conf, x, feature_map_size), ...
            conf.scales, 'UniformOutput', false);
    end

end

function [anchors, im_scale] = proposal_locate_anchors_single_scale(im_size, conf, target_scale, feature_map_size)
    if isempty(feature_map_size)
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        img_size = round(im_size * im_scale);
        output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
    else
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf.max_size);
        output_size = feature_map_size;
    end
    % 计算使用proposal_test.prototxt中RPN网络输出的feature map中每个像素点，对应于缩放后原图中的位置。即将feature map中的每个像素点位置×conf.feat_stride，即可映射回原图位置(缩放后的)。（感受野的概念）
    shift_x = [0:(output_size(2)-1)] * conf.feat_stride;
    shift_y = [0:(output_size(1)-1)] * conf.feat_stride;
    [shift_x, shift_y] = meshgrid(shift_x, shift_y); % 调试，理解meshgrid()函数功能，将shift_x，shift_y的变化情况弄明白
    
    %% added by lly on 2018.4.25 16:52  show default 9 anchors size -------
%     showDefaultAnchorSize(conf.anchors);
    % ---------------------------------------------------------------------
    
    %% （要多理解几遍）将feature map每个像素点映射回原图之后的位置，计算其在原图上的9个anchor(共有feature_map_size个位置)，即可得到9×feature_map_size个anchor
    % concat anchors as [channel, height, width], where channel is the fastest dimension.
    anchors = reshape(bsxfun(@plus, permute(conf.anchors, [1, 3, 2]), ...
        permute([shift_x(:), shift_y(:), shift_x(:), shift_y(:)], [3, 1, 2])), [], 4);
    
%   equals to  
%     anchors = arrayfun(@(x, y) single(bsxfun(@plus, conf.anchors, [x, y, x, y])), shift_x, shift_y, 'UniformOutput', false);
%     anchors = reshape(anchors, [], 1);
%     anchors = cat(1, anchors{:});

end

%% This function is used to show default 9 anchors size
function [] = showDefaultAnchorSize(boxes)
    figure(1);
    c=colormap(jet(length(boxes)));
    for i = 1 : length(boxes)
        rectangle('Position', [boxes(i,1), boxes(i,2), boxes(i,3) - boxes(i,1), boxes(i,4) - boxes(i,2)], 'EdgeColor', c(i,:), 'LineWidth', 2);
        pause(0.001);
        title(int2str(i));
    end
end