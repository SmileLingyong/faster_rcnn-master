function [] = testing() 
    classes = imdb.classes;
    rand_img_idx = randperm(num_images);
    aboxes_lly = aboxes;
    boxes_cell = cell(length(classes), 1);
    thres = 0.8;
    for ii = 1:10
        im_lly = imread(imdb.image_at(rand_img_idx(ii)));
        for jj = 1:length(boxes_cell)
            boxes_cell{jj} = aboxes_lly{jj, 1}{rand_img_idx(ii), 1};
            boxes_cell{jj} = boxes_cell{jj}(nms(boxes_cell{jj}, 0.3), :);

            I = boxes_cell{jj}(:, 5) > thres;
            boxes_cell{jj} = boxes_cell{jj}(I, :);
        end
        figure(ii);
        showboxes(im_lly, boxes_cell, classes, 'voc');
        pause(0.1);
    end
end