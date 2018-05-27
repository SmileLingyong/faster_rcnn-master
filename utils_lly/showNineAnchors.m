% Creation          :   11-Apr-2018 19:57
% Last Reversion    :   11-Apr-2018 19:57
% Author            :   Lingyong Smile {smilelingyong@163.com}
% File Type         :   Matlab
% 
% This function is used to show bounding boxes on a image.

function [] = showNineAnchors(im, boxes)
    figure(1);
    im = permute(im, [2, 1, 3, 4]);
    im_size = size(im);
    imshow(im);
%     imshow(rgb2gray(im));
    axis([-400 1200 -400 1000]);
    c=[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1];
    for i = 1 : length(boxes)   
        rectangle('Position', [boxes(i,1), boxes(i,2), boxes(i,3) - boxes(i,1) + 1, boxes(i,4) - boxes(i,2) + 1], 'EdgeColor', c(CalculateMod(i, 9),:), 'LineWidth', 2);
        label = sprintf('area:%dx%d scale:%s', boxes(i,3) - boxes(i,1) + 1, boxes(i,4) - boxes(i,2) + 1, calculateScale(boxes(i,3) - boxes(i,1) + 1,  boxes(i,4) - boxes(i,2) + 1));
%         label = sprintf('area:%dx%d', boxes(i,3) - boxes(i,1) + 1, boxes(i,4) - boxes(i,2) + 1);
        if  mod(i, 9)>=1 && mod(i, 9)<=3 && mod(i, 9)==3     % red, 2:1
            text(double(boxes(i,1)+2), double(boxes(i,2)), label, 'BackgroundColor', 'r');
        elseif mod(i, 9)>=4 && mod(i, 9)<=6  % green, 1:1
            text(double(boxes(i,1)+2), double(boxes(i,2)), label, 'BackgroundColor', 'w');
        elseif (mod(i, 9)>=7 || mod(i, 9)==0) &&  mod(i, 9)==0          % blue, 1:2
            text(double(boxes(i,1)+2), double(boxes(i,2)), label, 'BackgroundColor', 'b');
        end
        if mod(i, 9) == 0
            pause(0.5);
        end
    end
    title(sprintf('image size = %d x %d', im_size(1), im_size(2))); 
end

function scale = calculateScale(width, height)
    if round(width / height) == 2
        scale = '2:1';
    elseif width / height == 1
        scale = '1:1';
    elseif width / height == 0.5
        scale = '1:2';
    end
end

function result = CalculateMod(i, m)
    if mod(i, m) == 0
        result = m;
    else 
        result = mod(i, m);
    end
end