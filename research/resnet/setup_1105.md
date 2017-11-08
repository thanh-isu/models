11/05

*Setup environment for running resnet with CIFAR-10*

1. Download CIFAR-10 dataset
2. Extract it to cifar10 (Originally, named as cifar-10-batches-bin)
3. Installation of TensorFlow and Bazel
   +) Install Bazel: (Bazel is not available in RHEL package/repo)
   Download the bazel repo from https://copr.fedorainfracloud.org/coprs/vbatts/bazel/ for centos
   Add this *.repo to /etc/yum.repos.d
   Then,
   sudo yum install bazel
   +) Create virtual environment and install Tensorflow
    Use virtual environment in /local2/workspace/tensorflow/venv_gpu


4. Run bazel build
   +) Create an empty WORKSPACE file for bazel using
   ```shell
   > WORKSPACE
   ```

5. Training ResNet model:
Running command (inside Python virtual environment): 
python resnet_main.py --train_data_patch=cifar10/data_batch* --log_root=/tmp/resnet_model --train_dir=/tmp/resnet_model/train --dataset='cifar10' --num_gpus=0

6. Evaluate the model
# Evaluate the model.
# Avoid running on the same GPU as the training job at the same time,
# otherwise, you might run out of memory.
$ bazel-bin/resnet/resnet_main --eval_data_path=cifar10/test_batch.bin \
                               --log_root=/tmp/resnet_model_20_16x16 \
                               --eval_dir=/tmp/resnet_model_20_16x16/test \
                               --mode=eval \
                               --dataset='cifar10' \
                               --num_gpus=0


By default: number of residual units are 5 => network has 6n+2 = 32 layers and the input image is 32x32

Change num_residual_units to 3 and input image to 16x16. The log is stored in /tmp/resnet_model_20_16x16. Here is the model summary:

init/init_conv/DW (3x3x3x16, 432/432 params)
  logit/DW (64x10, 640/640 params)
  logit/biases (10, 10/10 params)
  unit_1_0/shared_activation/init_bn/beta (16, 16/16 params)
  unit_1_0/shared_activation/init_bn/gamma (16, 16/16 params)
  unit_1_0/sub1/conv1/DW (3x3x16x16, 2.30k/2.30k params)
  unit_1_0/sub2/bn2/beta (16, 16/16 params)
  unit_1_0/sub2/bn2/gamma (16, 16/16 params)
  unit_1_0/sub2/conv2/DW (3x3x16x16, 2.30k/2.30k params)
  unit_1_1/residual_only_activation/init_bn/beta (16, 16/16 params)
  unit_1_1/residual_only_activation/init_bn/gamma (16, 16/16 params)
  unit_1_1/sub1/conv1/DW (3x3x16x16, 2.30k/2.30k params)
  unit_1_1/sub2/bn2/beta (16, 16/16 params)
  unit_1_1/sub2/bn2/gamma (16, 16/16 params)
  unit_1_1/sub2/conv2/DW (3x3x16x16, 2.30k/2.30k params)
  unit_1_2/residual_only_activation/init_bn/beta (16, 16/16 params)
  unit_1_2/residual_only_activation/init_bn/gamma (16, 16/16 params)
  unit_1_2/sub1/conv1/DW (3x3x16x16, 2.30k/2.30k params)
  unit_1_2/sub2/bn2/beta (16, 16/16 params)
  unit_1_2/sub2/bn2/gamma (16, 16/16 params)
  unit_1_2/sub2/conv2/DW (3x3x16x16, 2.30k/2.30k params)
  unit_2_0/residual_only_activation/init_bn/beta (16, 16/16 params)
  unit_2_0/residual_only_activation/init_bn/gamma (16, 16/16 params)
  unit_2_0/sub1/conv1/DW (3x3x16x32, 4.61k/4.61k params)
  unit_2_0/sub2/bn2/beta (32, 32/32 params)
  unit_2_0/sub2/bn2/gamma (32, 32/32 params)
  unit_2_0/sub2/conv2/DW (3x3x32x32, 9.22k/9.22k params)
  unit_2_1/residual_only_activation/init_bn/beta (32, 32/32 params)
  unit_2_1/residual_only_activation/init_bn/gamma (32, 32/32 params)
  unit_2_1/sub1/conv1/DW (3x3x32x32, 9.22k/9.22k params)
  unit_2_1/sub2/bn2/beta (32, 32/32 params)
  unit_2_1/sub2/bn2/gamma (32, 32/32 params)
  unit_2_1/sub2/conv2/DW (3x3x32x32, 9.22k/9.22k params)
  unit_2_2/residual_only_activation/init_bn/beta (32, 32/32 params)
  unit_2_2/residual_only_activation/init_bn/gamma (32, 32/32 params)
  unit_2_2/sub1/conv1/DW (3x3x32x32, 9.22k/9.22k params)
  unit_2_2/sub2/bn2/beta (32, 32/32 params)
  unit_2_2/sub2/bn2/gamma (32, 32/32 params)
  unit_2_2/sub2/conv2/DW (3x3x32x32, 9.22k/9.22k params)
  unit_3_0/residual_only_activation/init_bn/beta (32, 32/32 params)
  unit_3_0/residual_only_activation/init_bn/gamma (32, 32/32 params)
  unit_3_0/sub1/conv1/DW (3x3x32x64, 18.43k/18.43k params)
  unit_3_0/sub2/bn2/beta (64, 64/64 params)
  unit_3_0/sub2/bn2/gamma (64, 64/64 params)
  unit_3_0/sub2/conv2/DW (3x3x64x64, 36.86k/36.86k params)
  unit_3_1/residual_only_activation/init_bn/beta (64, 64/64 params)
  unit_3_1/residual_only_activation/init_bn/gamma (64, 64/64 params)
  unit_3_1/sub1/conv1/DW (3x3x64x64, 36.86k/36.86k params)
  unit_3_1/sub2/bn2/beta (64, 64/64 params)
  unit_3_1/sub2/bn2/gamma (64, 64/64 params)
  unit_3_1/sub2/conv2/DW (3x3x64x64, 36.86k/36.86k params)
  unit_3_2/residual_only_activation/init_bn/beta (64, 64/64 params)
  unit_3_2/residual_only_activation/init_bn/gamma (64, 64/64 params)
  unit_3_2/sub1/conv1/DW (3x3x64x64, 36.86k/36.86k params)
  unit_3_2/sub2/bn2/beta (64, 64/64 params)
  unit_3_2/sub2/bn2/gamma (64, 64/64 params)
  unit_3_2/sub2/conv2/DW (3x3x64x64, 36.86k/36.86k params)
  unit_last/final_bn/beta (64, 64/64 params)
  unit_last/final_bn/gamma (64, 64/64 params)

======================End of Report==========================
total_params: 269722
Parsing GraphDef...
Parsing RunMetadata...
Parsing OpLog...
Preparing Views...

=========================Options=============================
-max_depth                  10000
-min_bytes                  0
-min_micros                 0
-min_params                 0
-min_float_ops              1
-device_regexes             .*
-order_by                   float_ops
-account_type_regexes       .*
-start_name_regexes         .*
-trim_name_regexes          
-show_name_regexes          .*
-hide_name_regexes          
-account_displayed_op_only  true
-select                     float_ops
-viz                        false
-dump_to_file               

==================Model Analysis Report======================
_TFProfRoot (0/10.38b flops)
  unit_2_0/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_3_2/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_3_2/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_3_1/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_3_1/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_3_0/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_2_2/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_2_2/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_2_1/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_2_1/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_1_2/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_1_2/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_1_1/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_1_1/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_1_0/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_1_0/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_2_0/sub1/conv1/Conv2D (301.99m/301.99m flops)
  unit_3_0/sub1/conv1/Conv2D (301.99m/301.99m flops)
  init/init_conv/Conv2D (113.25m/113.25m flops)
  logit/xw_plus_b (1.28k/165.12k flops)
    logit/xw_plus_b/MatMul (163.84k/163.84k flops)
  gradients/logit/xw_plus_b/MatMul_grad/MatMul (163.84k/163.84k flops)
  gradients/logit/xw_plus_b/MatMul_grad/MatMul_1 (163.84k/163.84k flops)


