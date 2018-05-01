% Creation              :   05-Mar-2018 15:26
% Last Reversion        :   05-Mar-2018 15:26
% Author                :   Lingyong Smile{smilelingyong@163.com}
% File Type             :   Matlab
% 
% This function is used to show partial weights map.
% ----------------------------------------------------------------
% Smeil Lingyong @ 2018 
% ----------------------------------------------------------------

%% 调用方法
% conv_layer_name = 'conv2';  % 查看conv2的卷积核对应的权重
% channel_num = 48;  % 查看第48通道的
% weight_partial_visual(net, conv_layer_name, channel_num);

function [] = weight_partial_visual(net, conv_layer_name, channel_num)
    layers = net.layer_names;
    conv_num = conv_layer_name(end);
    conv_layers = [];
    for i = 1:length(layers)
        if(strcmp(layers{i}(1:3), 'con'))  % 仅仅卷积核能获取到权重
            conv_layers = [conv_layers; layers{i}];
        end
    end
    
    % 得到当前conv层的weight 和 bias,并对weight进行归一化处理
    w = net.layers(conv_layer_name).params(1).get_data();  % 存储权重
    b = net.layers(conv_layer_name).params(2).get_data();  % 存储偏置
    minval = min(min(min(min(w))));
    maxval = max(max(max(max(w))));
    w = (w - minval) / maxval * 255;  % 归一化处理
    
    weight = w(:, :, channel_num, :);  % 注意调试看一下，当channel_num!=1 的时候，比如channel_num=2的时候，weight中存储的内容，这样你才能明白下面的weight(:, :, :, kernel_num_idx); 第三维度为何也是：
    [h_size, w_size, input_num, kernel_num] = size(weight);
    row = ceil(sqrt(kernel_num));
    col = row;
    weight_map = zeros(row*h_size, col*w_size);
    kernel_num_idx = 1;
    for i = 0:row-1
        for j = 0:col-1
            if kernel_num_idx <= kernel_num
                weight_map(i*h_size+1:(i+1)*h_size, j*w_size+1:(j+1)*w_size) = weight(:, :, :, kernel_num_idx);  
                kernel_num_idx = kernel_num_idx + 1;
            end
        end
    end
    
    % show weight map
    imshow(uint8(weight_map));
    str = strcat('weight map num:', num2str(kernel_num));
    title(str);

end