% Creation          :   11-Apr-2018 19:57
% Last Reversion    :   11-Apr-2018 19:57
% Author            :   Lingyong Smile {smilelingyong@163.com}
% File Type         :   Matlab
% 
% This function is used to show bounding boxes on a image.

function [] = showSelectiveSearchBBox(im, boxes)
    figure(1);
    imshow(im); 
    c=colormap(jet(length(boxes)));
    for i = 1 : length(boxes)
        rectangle('Position', [boxes(i,1), boxes(i,2), boxes(i,3) - boxes(i,1), boxes(i,4) - boxes(i,2)], 'EdgeColor', c(i,:), 'LineWidth', 2);
        pause(0.001);
        title(int2str(i));
    end
end