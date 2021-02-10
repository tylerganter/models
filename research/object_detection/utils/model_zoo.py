"""Utilities for downloading and managing cached models"""
import os
import re

import tensorflow as tf


# the base URL for *most* models
BASE_URL = "http://download.tensorflow.org/models/object_detection/"

# a map from tensorflow version + model name to model URL
MODEL_URL_MAP = {
    "tf1": {
        "ssd_mobilenet_v1_coco":                                     "ssd_mobilenet_v1_coco_2018_01_28.tar.gz",
        "ssd_mobilenet_v1_0.75_depth_coco":                          "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz",
        "ssd_mobilenet_v1_quantized_coco":                           "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
        "ssd_mobilenet_v1_0.75_depth_quantized_coco":                "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
        "ssd_mobilenet_v1_ppn_coco":                                 "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz",
        "ssd_mobilenet_v1_fpn_coco":                                 "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz",
        "ssd_resnet_50_fpn_coco":                                    "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz",
        "ssd_mobilenet_v2_coco":                                     "ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
        "ssd_mobilenet_v2_quantized_coco":                           "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz",
        "ssdlite_mobilenet_v2_coco":                                 "ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz",
        "ssd_inception_v2_coco":                                     "ssd_inception_v2_coco_2018_01_28.tar.gz",
        "faster_rcnn_inception_v2_coco":                             "faster_rcnn_inception_v2_coco_2018_01_28.tar.gz",
        "faster_rcnn_resnet50_coco":                                 "faster_rcnn_resnet50_coco_2018_01_28.tar.gz",
        "faster_rcnn_resnet50_lowproposals_coco":                    "faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz",
        "rfcn_resnet101_coco":                                       "rfcn_resnet101_coco_2018_01_28.tar.gz",
        "faster_rcnn_resnet101_coco":                                "faster_rcnn_resnet101_coco_2018_01_28.tar.gz",
        "faster_rcnn_resnet101_lowproposals_coco":                   "faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz",
        "faster_rcnn_inception_resnet_v2_atrous_coco":               "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz",
        "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco":  "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz",
        "faster_rcnn_nas":                                           "faster_rcnn_nas_coco_2018_01_28.tar.gz",
        "faster_rcnn_nas_lowproposals_coco":                         "faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz",
        "mask_rcnn_inception_resnet_v2_atrous_coco":                 "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz",
        "mask_rcnn_inception_v2_coco":                               "mask_rcnn_inception_v2_coco_2018_01_28.tar.gz",
        "mask_rcnn_resnet101_atrous_coco":                           "mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz",
        "mask_rcnn_resnet50_atrous_coco":                            "mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz",
        "ssd_mobiledet_cpu_coco":                                    "ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz",
        "ssd_mobilenet_v2_mnasfpn_coco":                             "ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tar.gz",
        "ssd_mobilenet_v3_large_coco":                               "ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz",
        "ssd_mobilenet_v3_small_coco":                               "ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz",
        "ssd_mobiledet_edgetpu_coco":                                "ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz",
        "ssd_mobilenet_edgetpu_coco":                                "https://storage.cloud.google.com/mobilenet_edgetpu/checkpoints/ssdlite_mobilenet_edgetpu_coco_quant.tar.gz",
        "ssd_mobiledet_dsp_coco":                                    "ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tar.gz",
        "faster_rcnn_resnet101_kitti":                               "faster_rcnn_resnet101_kitti_2018_01_28.tar.gz",
        "faster_rcnn_inception_resnet_v2_atrous_oidv2":              "faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz",
        "faster_rcnn_inception_resnet_v2_atrous_lowproposals_oidv2": "faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz",
        "facessd_mobilenet_v2_quantized_open_image_v4":              "facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz",
        "faster_rcnn_inception_resnet_v2_atrous_oidv4":              "faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz",
        "ssd_mobilenetv2_oidv4":                                     "ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz",
        "ssd_resnet_101_fpn_oidv4":                                  "ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz",
        "faster_rcnn_resnet101_fgvc":                                "faster_rcnn_resnet101_fgvc_2018_07_19.tar.gz",
        "faster_rcnn_resnet50_fgvc":                                 "faster_rcnn_resnet50_fgvc_2018_07_19.tar.gz",
        "faster_rcnn_resnet101_ava_v2.1":                            "faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz",
        "faster_rcnn_resnet101_snapshot_serengeti":                  "faster_rcnn_resnet101_snapshot_serengeti_2020_06_10.tar.gz",
        "context_rcnn_resnet101_snapshot_serengeti":                 "context_rcnn_resnet101_snapshot_serengeti_2020_06_10.tar.gz",
    },
    "tf2": {
        "CenterNet HourGlass104 512x512":                "tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz",
        "CenterNet HourGlass104 Keypoints 512x512":      "tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz",
        "CenterNet HourGlass104 1024x1024":              "tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz",
        "CenterNet HourGlass104 Keypoints 1024x1024":    "tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz",
        "CenterNet Resnet50 V1 FPN 512x512":             "tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz",
        "CenterNet Resnet50 V1 FPN Keypoints 512x512":   "tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz",
        "CenterNet Resnet101 V1 FPN 512x512":            "tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz",
        "CenterNet Resnet50 V2 512x512":                 "tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz",
        "CenterNet Resnet50 V2 Keypoints 512x512":       "tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz",
        "EfficientDet D0 512x512":                       "tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz",
        "EfficientDet D1 640x640":                       "tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz",
        "EfficientDet D2 768x768":                       "tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz",
        "EfficientDet D3 896x896":                       "tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz",
        "EfficientDet D4 1024x1024":                     "tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz",
        "EfficientDet D5 1280x1280":                     "tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz",
        "EfficientDet D6 1280x1280":                     "tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz",
        "EfficientDet D7 1536x1536":                     "tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz",
        "SSD MobileNet v2 320x320":                      "tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
        "SSD MobileNet V1 FPN 640x640":                  "tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "SSD MobileNet V2 FPNLite 320x320":              "tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz",
        "SSD MobileNet V2 FPNLite 640x640":              "tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz",
        "SSD ResNet50 V1 FPN 640x640 (RetinaNet50)":     "tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)":   "tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
        "SSD ResNet101 V1 FPN 640x640 (RetinaNet101)":   "tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)": "tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
        "SSD ResNet152 V1 FPN 640x640 (RetinaNet152)":   "tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)": "tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
        "Faster R-CNN ResNet50 V1 640x640":              "tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz",
        "Faster R-CNN ResNet50 V1 1024x1024":            "tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz",
        "Faster R-CNN ResNet50 V1 800x1333":             "tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz",
        "Faster R-CNN ResNet101 V1 640x640":             "tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz",
        "Faster R-CNN ResNet101 V1 1024x1024":           "tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz",
        "Faster R-CNN ResNet101 V1 800x1333":            "tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz",
        "Faster R-CNN ResNet152 V1 640x640":             "tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz",
        "Faster R-CNN ResNet152 V1 1024x1024":           "tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz",
        "Faster R-CNN ResNet152 V1 800x1333":            "tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz",
        "Faster R-CNN Inception ResNet V2 640x640":      "tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz",
        "Faster R-CNN Inception ResNet V2 1024x1024":    "tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz",
        "Mask R-CNN Inception ResNet V2 1024x1024":      "tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz",
        "ExtremeNet":                                    "tf2/20200711/extremenet.tar.gz",
    },
}


def load_pretrained_model_dir(
    model_name, tf_version=1, cache_subdir="models/object_detection", **kwargs
):
    """Provides the directory to a pretrained model and downloads and untars
    the model if necessary.

    Args:
        model_name: the model name, which should be a key in the
            `MODEL_MAP` for the given TF version
        tf_version: the integer tensorflow version
        cache_subdir: subdirectory under the Keras cache dir where the model is
          saved. If an absolute path `/path/to/folder` is
          specified the file will be saved at that location.
        kwargs: passed to `tf.keras.utils.get_file()`

    Returns:
         model_dir: the path to the untared model directory
    """
    model_map = MODEL_URL_MAP[f"tf{tf_version}"]

    try:
        model_relative_url = model_map[model_name]
    except KeyError:
        raise KeyError(f"Invalid model name: {model_name} for TF{tf_version}")

    if _is_url(model_relative_url):
        origin = model_relative_url
    else:
        origin = BASE_URL + model_relative_url

    # use the basename of the tarfile as the `fname`
    model_key = os.path.basename(origin)[: -len(".tar.gz")]

    return tf.keras.utils.get_file(
        fname=model_key, cache_subdir=cache_subdir, origin=origin, untar=True, **kwargs
    )


def _is_url(s):
    """Check if the string is a URL"""
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, s) is not None
