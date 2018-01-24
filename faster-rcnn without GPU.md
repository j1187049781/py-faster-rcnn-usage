 ## 1 NOT_IMPLEMENTED issue
py-faster-rcnn在测试模型的时候，可以选择使用cpu mode或者gpu mode，但是如果使用该框架训练自己的模型，就只能使用gpu了。应该是作者考虑训练速度的原因，对roi_pooling_layer和smooth_L1_loss_layer只使用和提供了gpu版本的代码.
这两个文件在py-fast-rcnn/caffe-fast-rcnn/src/caffe/layers 。打开这两个文件，可以看到smooth_L1_loss_layer.cpp中forward和backward处都是NOT_IMPLEMENTED。 所以如果没有一块满足性能的GPU就做不了训练了。
下边是我对这两个文件的修改，实现了CPU版本的函数，如有错误，欢迎指正交流。另外，在[github](https://github.com/neuleaf/faster-rcnn-cpu)上也可以找到这两个文件。使用时，直接替换原文件,重新make即可
 ## 2 TypeError: slice indices must be integers or None or have an __index__ method
 numpy的版本问题:

修改 /home/lzx/py-faster-rcnn/lib/rpn/proposal_target_layer.py，转到123行：

''' python

    for ind in inds:  
    cls = clss[ind]  
    start = 4 * cls  
    end = start + 4  
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS  
    return bbox_targets, bbox_inside_weights  
'''  
这里的ind，start，end都是 numpy.int 类型，这种类型的数据不能作为索引，所以必须对其进行强制类型转换，转化结果如下：
''' python

    for ind in inds:  
    ind = int(ind)  
    cls = clss[ind]  
    start = int(4 * cls)  
    end = int(start + 4)  
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS  
    return bbox_targets, bbox_inside_weights  
'''
## 3 AttributeError: 'module' object has no attribute ‘text_format'
解决方法：在/home/xxx/py-faster-rcnn/lib/fast_rcnn/train.py的头文件导入部分加上 ：import google.protobuf.text_format

## 4 TypeError: 'numpy.float64' object cannot be interpreted as an index
这个问题是因为我用的numpy版本太高了,　最简单的方法是直接改版本　sudo python2.7 /usr/local/bin/pip install -U numpy==1.11.0;  
或者修改如下几个地方的code
　　1) /home/xxx/py-faster-rcnn/lib/roi_data_layer/minibatch.py

　　将第26行：fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
　　改为：fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)


　　2) /home/xxx/py-faster-rcnn/lib/datasets/ds_utils.py
　　将第12行：hashes = np.round(boxes * scale).dot(v)
　　改为：hashes = np.round(boxes * scale).dot(v).astype(np.int)


　　3) /home/xxx/py-faster-rcnn/lib/fast_rcnn/test.py
　　将第129行： hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
　　改为： hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v).astype(np.int)


　　4) /home/xxx/py-faster-rcnn/lib/rpn/proposal_target_layer.py

　　将第60行：fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

　　改为：fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
