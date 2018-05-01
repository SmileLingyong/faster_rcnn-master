% Creation              :   05-Mar-2018 15:26
% Last Reversion        :   05-Mar-2018 15:26
% Author                :   Lingyong Smile{smilelingyong@163.com}
% File Type             :   Matlab
% 
% This function is used to show full weights map.
% ----------------------------------------------------------------
% Smeil Lingyong @ 2018 
% ----------------------------------------------------------------
function [] = weight_full_visual(net, conv_layer_name)
    layers = net.layer_names;
    conv_num = conv_layer_name(end);
    conv_layers = [];
    for i = 1:length(conv_layer_name)
        if(strcmp(layers{i}(1:3), 'con'))
            conv_layers = [conv_layers; layers{i}];
        end
    end
    
    % 得到当前conv层的 weight 和 bias,并对weight进行归一化处理
    w = net.layers(conv_layer_name).params(1).get_data();
    b = net.layers(conv_layer_name).params(2).get_data();
    minval = min(min(min(min(w))));
    maxval = max(max(max(max(w))));
    w = (w - minval) / maxval * 256;
    
    [h_size, w_size, input_num, kernel_num] = size(w);
    weight_map = zeros(h_size*input_num, w_size*kernel_num);
    for i = 0:input_num-1
        for j = 0:kernel_num-1
            weight_map(i*h_size+1:(i+1)*h_size, j*w_size+1:(j+1)*w_size) = w(:, :, i+1, j+1);
        end
    end
    
    % show weight map
    imshow(uint8(weight_map));
    str = strcat('weight map num:', num2str(kernel_num));
    title(str);
end

