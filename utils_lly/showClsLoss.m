% Creation              :       29-Apr-2018 16:34
% Last Reversion        :       29-Apr-2018 16:34
% Author                :       Lingyong Smile{smilelingyong@163.com}
% File Type             :       Matlab
%
% This function is used to plot classification loss curve.
% -------------------------------------------------------------------------
% Crop right @ Lingyong Smile 2008

function [] = showClsLoss(train_results)
    loss_cls = train_results.loss_cls.data;
    X = 1:numel(loss_cls);
    line(X, loss_cls);
    title(sprintf('Training Loss, iter-%s', num2str(length(X))));
    pause(0.01);
end
