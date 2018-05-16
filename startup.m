function startup()
% startup()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    curdir = fileparts(mfilename('fullpath'));
    %% uncommented by lly on 2018.4.17 15:05
%     addpath(genpath(fullfile(curdir, 'utils')));
%     addpath(genpath(fullfile(curdir, 'functions')));
%     addpath(genpath(fullfile(curdir, 'bin')));
%     addpath(genpath(fullfile(curdir, 'experiments')));
%     addpath(genpath(fullfile(curdir, 'imdb')));

    mkdir_if_missing(fullfile(curdir, 'datasets'));

    mkdir_if_missing(fullfile(curdir, 'external'));

    caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab');
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));

    mkdir_if_missing(fullfile(curdir, 'imdb', 'cache'));

    mkdir_if_missing(fullfile(curdir, 'output'));

    mkdir_if_missing(fullfile(curdir, 'models'));

    fprintf('fast_rcnn startup done\n');
end
