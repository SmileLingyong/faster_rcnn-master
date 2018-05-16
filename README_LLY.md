# Some Notes

## caffe编译

- 首先，自己下载了官网faster-rcnn 中的caffe版本，将其放置到`external/`下，则目录结构为`external/caffe`

- 然后，开始编译该版本的caffe，自己直接复制了原先编译好的`caffe-master`中的`Makefile.config`到该目录，最为编译的配置文件，并注释掉`# USE_CUDNN := 1` ，即不使用cudnn编译 (因为该版本的caffe不支持cudnn5，编译的时候会报错，所以自己就没有用cudnn)。

- 这里编译的时候会出现一个问题，然后参考了[这篇博客](https://blog.csdn.net/Xiongchao99/article/details/79100787) 修改后编译成功。

  **自己博客中记录了可能解决这个问题的办法，自己还没有实现**

  **自己将这份编译好的caffe-faster-rcnn-cpu版本备份了一份到`/home/lly/work/sina/backup`中**

## dataset准备

- 下载了`selective_search_data`  然后放置到新建的`data` 目录
- `datasets` 中，目录结构如下所示

```shell
lly@lly:~/work/sina/RCNN/faster_rcnn-master/datasets$ tree -L 2
.
├── faster_rcnn_final_model.zip  	# 将这个zip直接在根目录解压即可，会得到out目录以及其他文件
├── faster_rcnn_logs.zip
├── models
│   ├── fast_rcnn_prototxts
│   ├── pre_trained_models
│   └── rpn_prototxts
├── model_VGG16.zip    				# VGG16训练好了的model，使用下面的语句解压即可	
├── model_ZF.zip					# ZF训练好了的model	
├── proposals_faster_rcnn_VOC0712_vgg_16layers.zip
├── README.md
├── VOCdevkit
│   ├── create_segmentations_from_detections.m
│   ├── devkit_doc.pdf
│   ├── example_classifier.m
│   ├── example_detector.m
│   ├── example_layout.m
│   ├── example_segmenter.m
│   ├── local
│   ├── results
│   ├── viewanno.m
│   ├── viewdet.m
│   ├── VOC2007
│   ├── VOC2012
│   ├── VOCcode
│   └── VOCdevkit -> VOCdevkit
├── VOCdevkit_08-Jun-2007.tar
├── VOCdevkit2007 -> VOCdevkit
├── VOCdevkit2012 -> VOCdevkit
├── VOCtest_06-Nov-2007.tar
├── VOCtrainval_06-Nov-2007.tar
└── VOCtrainval_11-May-2012.tar
```

解压上面的 `model_VGG16.zip` 和 `model_ZF.zip` 以及那几个`VOCdevkit` 数据集的 命令

```shell
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

Create symlinks for the PASCAL VOC dataset

```shell
ln -s VOCdevkit VOCdevkit2007
ln -s VOCdevkit VOCdevkit2012
```

这里是参考了 [Fast-rcnn的README.md](https://github.com/rbgirshick/fast-rcnn)





#### 2018.4.10  20:26 -------------------------------------------

找到了一份人家改好了的`caffe_fast_rcnn-master-unofficial` 可以编译成功.(自己将其放到 `external/`目录之下)  https://github.com/owphoo/caffe_fast_rcnn







