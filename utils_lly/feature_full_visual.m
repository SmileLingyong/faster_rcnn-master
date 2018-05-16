% Creation              :   27-Feb-2018 14:47
% Last Reversion        :   27-Feb-2018 14:47
% Author                :   Lingyong Smile{smilelingyong@163.com}
% File Type             :   Matlab
% 
% This function is used to show full feature.
% 这一部分可视化每一张输入图片在指定卷积层的特征图，按照每一行为存储图片的特征图为图例。
% ----------------------------------------------------------------
% Smeil Lingyong @ 2018 
% ----------------------------------------------------------------

%% 调用方法
% blob_name = 'conv1'; % 哪一层的特征图, conv1层
% feature_partvisual(net, blob_name); 


function [] = feature_full_visual(net, blob_name)
blob = net.blobs(blob_name).get_data(); % 获取指定层的blob
[h_size, w_size, num_output, crop] = size(blob); % 获取特征图大小，长*宽*卷积核个数*通道数
row = crop;         % 行数
col = num_output;   % 列数
feature_map = zeros(row*h_size, col*w_size);
for i = 0:row-1
    for j = 0:col-1
        % first method: using mapminmax() function. 循环的过程实际上是将每个num_output一行一行的存储到feature_map中  (!!!注意:因为MATLAB是默认列优先，所以第一位 i*h_size+1:(i+1)*h_size 表示的是表示的是当前num_output高度所在的位置,自己调试运行一下,画个图就可以看出feature中内容是如何变化的)
%         feature_map(i*h_size+1:(i+1)*h_size, j*w_size+1:(j+1)*w_size) = (mapminmax(blob(:, :, j+1, i+1), 0, 1) * 255)';
        
        %% second method
        w = blob(:, :, j+1, i+1)';
        w = w - min(min(w));
        w = w / max(max(w))*255;
        feature_map(i*h_size+1:(i+1)*h_size, j*w_size+1:(j+1)*w_size) = w;
    end
end
imshow(uint8(feature_map));
str = strcat('feature map num:', num2str(num_output));
title(str);
end