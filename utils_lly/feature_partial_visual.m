% Creation              :   27-Feb-2018 10:19
% Last Reversion        :   27-Feb-2018 10:19
% Author                :   Lingyong Smile{smilelingyong@163.com}
% File Type             :   Matlab
% 
% This function is used to show partial feature.
% 这一部分针对指定的第crop_num张图像在第blob_name层进行可视化。注意，这一部分的可视化包含池化层等。
% ----------------------------------------------------------------
% Smeil Lingyong @ 2018 
% ----------------------------------------------------------------


%% 调用方法
% blob_name = 'conv1'; % 哪一层的特征图, conv1层
% crop_num = 1;        % 第几个crop的特征图, 如第一个crop的特征图
% feature_partvisual(net, blob_name, crop_num); 


function [] = feature_partial_visual(net, blob_name, crop_num)
blob = net.blobs(blob_name).get_data(); % 获取指定层的blob
[h_size, w_size, num_output, crop] = size(blob); % 获取特征图大小，长*宽*卷积核个数*通道数
row = ceil(sqrt(num_output));  % 将这些卷积核展示出来所需要的行数
col = row;  % 列数
feature_map = zeros(row*h_size, col*w_size); 
num_output_idx = 1;
for i = 0:row-1
    for j = 0:col-1
        if num_output_idx <= num_output
            % 循环的过程实际上是将每个num_output一行一行的存储到feature_map中  (!!!注意:因为MATLAB是默认列优先，所以第一位 i*h_size+1:(i+1)*h_size 表示的是表示的是当前num_output高度所在的位置,自己调试运行一下,画个图就可以看出feature中内容是如何变化的)
            %% first method: using mapminmax() function to normalization. reference: http://blog.csdn.net/hqh45/article/details/42965481
%             feature_map(i*h_size+1:(i+1)*h_size, j*w_size+1:(j+1)*w_size) = (mapminmax(blob(:, :, num_output_idx, crop_num), 0, 1)*255)';
            
            %% second method: 
            w = blob(:, :, num_output_idx, crop_num)';
            w = w - min(min(w));      % 保证为非负数
            w = w / max(max(w))*225;  % 归一化
            feature_map(i*h_size+1:(i+1)*h_size, j*w_size+1:(j+1)*w_size) = w;
            
            num_output_idx = num_output_idx + 1;
        end
    end
end
imshow(uint8(feature_map));
str = strcat('feature map num:', num2str(num_output));
title(str);
end
