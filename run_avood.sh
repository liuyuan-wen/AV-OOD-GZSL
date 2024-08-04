#!/bin/bash

# simply set any run_* to 1 and ignore the rest to run the corresponding experiment
run_activity=1
run_ucf=
run_vgg=

max_concurrent=10
current_concurrent=0

root_set='avgzsl_benchmark_datasets'

# 0:l+l, 1:lpos, 2:lpos+ltrip, 3:lpos+lreg, 4:lpos+lrec
loss_mode=0

scan_pretrain=False
# scan_pretrain=True

# r_enc, r_proj, r_dec
r_enc_activity=0.2
r_proj_activity=0.3
r_dec_activity=0.2
seen_contrasive_ratio_activity=0

r_enc_ucf=0.5
r_proj_ucf=0.4
r_dec_ucf=0.4
seen_contrasive_ratio_ucf=0

r_enc_vgg=0.3
r_proj_vgg=0.1
r_dec_vgg=0.2
seen_contrasive_ratio_vgg=0

epochs_activity=5
epochs_ucf=5
epochs_vgg=5

n_batches_activity=500
n_batches_ucf=500
n_batches_vgg=500

bs_activity=256 # 256
bs_ucf=112  # 256
bs_vgg=256  # 64

lr_activity=0.0005 # 0.001
lr_ucf=0.0024
lr_vgg=0.0005

lr_clss_activity=0.008
batch_size_clss_activity=128
syn_num_ood_activity=1000
lr_ood_activity=0.005
batch_size_ood_activity=64

lr_clss_ucf=0.0006
batch_size_clss_ucf=32
syn_num_ood_ucf=50
lr_ood_ucf=0.0009
batch_size_ood_ucf=16

lr_clss_vgg=0.008
batch_size_clss_vgg=1024
syn_num_ood_vgg=50
lr_ood_vgg=0.001
batch_size_ood_vgg=6900

num_workers=8

# run_file_path_head="test_l+l"
exp_name_suffix='main'

run_pid_activity=0
run_pid_ucf=0
run_pid_vgg=0



if [ "$loss_mode" -eq 0 ]; then
    ltrip_neg=True
    lreg_neg=True
    lrec_neg=True
    run_file_path_head="test_l+l"
elif [ "$loss_mode" -eq 1 ]; then
    ltrip_neg=False
    lreg_neg=False
    lrec_neg=False
    run_file_path_head="test_lpos"
elif [ "$loss_mode" -eq 2 ]; then
    ltrip_neg=True
    lreg_neg=False
    lrec_neg=False
    run_file_path_head="test_lpos+ltrip"
elif [ "$loss_mode" -eq 3 ]; then
    ltrip_neg=False
    lreg_neg=True
    lrec_neg=False
    run_file_path_head="test_lpos+lreg"
elif [ "$loss_mode" -eq 4 ]; then
    ltrip_neg=False
    lreg_neg=False
    lrec_neg=True
    run_file_path_head="test_lpos+lrec"
fi


if [ $run_activity = 1 ]; then
    if [ $seen_contrasive_ratio_activity = 0 ]; then
        run_file_path=""$run_file_path_head"_"$r_enc_activity"_"$r_proj_activity"_"$r_dec_activity""
    else
        run_file_path=""$run_file_path_head"_"$r_enc_activity"_"$r_proj_activity"_"$r_dec_activity"_"$seen_contrasive_ratio_activity""
    fi
    exp_name="activity_"$exp_name_suffix""
    rm -r $run_file_path/$exp_name

    sed -i "/^[^ ]/,/^\s*$/{
        s/run_file_path: .*/run_file_path: $run_file_path/
        s/exp_name: .*/exp_name: $exp_name/
        }" config/activity_test.yaml

    sed -i "/^dataset:/,/^[^ ]/{
        s/root_set: .*/root_set: $root_set/
        }" config/activity_test.yaml

    sed -i "/^gen:/,/^[^ ]/{
        s/epochs: .*/epochs: $epochs_activity/
        }" config/activity_test.yaml

    sed -i "/^clss:/,/^[^ ]/{
        s/epochs: .*/epochs: 200/
        s/lr: .*/lr: $lr_clss_activity/
        s/bs: .*/bs: $batch_size_clss_activity/
        }" config/activity_test.yaml

    sed -i "/^clsu:/,/^[^ ]/{
        s/epochs: .*/epochs: 50/
        s/num_workers: .*/num_workers: $num_workers/
        s/bs: .*/bs: $bs_activity/
        s/n_batches: .*/n_batches: $n_batches_activity/
        s/lr: .*/lr: $lr_activity/
        s/r_enc: .*/r_enc: $r_enc_activity/
        s/r_proj: .*/r_proj: $r_proj_activity/
        s/r_dec: .*/r_dec: $r_dec_activity/
        s/ltrip_neg: .*/ltrip_neg: $ltrip_neg/
        s/lreg_neg: .*/lreg_neg: $lreg_neg/
        s/lrec_neg: .*/lrec_neg: $lrec_neg/
        s/scan_pretrain: .*/scan_pretrain: $scan_pretrain/
        }" config/activity_test.yaml

    sed -i "/^ood:/,/^[^ ]/{
        s/epochs: .*/epochs: 80/
        s/lr: .*/lr: $lr_ood_activity/
        s/bs: .*/bs: $batch_size_ood_activity/
        s/syn_num: .*/syn_num: $syn_num_ood_activity/
        }" config/activity_test.yaml

    sed -i "s/seen_contrasive_ratio = .*/seen_contrasive_ratio = $seen_contrasive_ratio_activity/" src/global_var.py
    python main.py config/activity_test.yaml
    run_pid_activity=$!
    sleep 5
fi

if [ $run_ucf = 1 ]; then
    if [ $seen_contrasive_ratio_ucf = 0 ]; then
        run_file_path=""$run_file_path_head"_"$r_enc_ucf"_"$r_proj_ucf"_"$r_dec_ucf""
    else
        run_file_path=""$run_file_path_head"_"$r_enc_ucf"_"$r_proj_ucf"_"$r_dec_ucf"_"$seen_contrasive_ratio_ucf""
    fi
    exp_name="ucf_"$exp_name_suffix""
    rm -r $run_file_path/$exp_name

    sed -i "/^[^ ]/,/^\s*$/{
        s/run_file_path: .*/run_file_path: $run_file_path/
        s/exp_name: .*/exp_name: $exp_name/
        }" config/ucf_test.yaml

    sed -i "/^dataset:/,/^[^ ]/{
        s/root_set: .*/root_set: $root_set/
        }" config/ucf_test.yaml

    sed -i "/^gen:/,/^[^ ]/{
        s/epochs: .*/epochs: $epochs_ucf/
        }" config/ucf_test.yaml

    sed -i "/^clss:/,/^[^ ]/{
        s/lr: .*/lr: $lr_clss_ucf/
        s/bs: .*/bs: $batch_size_clss_ucf/
        }" config/ucf_test.yaml

    sed -i "/^clsu:/,/^[^ ]/{
        s/num_workers: .*/num_workers: $num_workers/
        s/bs: .*/bs: $bs_ucf/
        s/n_batches: .*/n_batches: $n_batches_ucf/
        s/lr: .*/lr: $lr_ucf/
        s/r_enc: .*/r_enc: $r_enc_ucf/
        s/r_proj: .*/r_proj: $r_proj_ucf/
        s/r_dec: .*/r_dec: $r_dec_ucf/
        s/ltrip_neg: .*/ltrip_neg: $ltrip_neg/
        s/lreg_neg: .*/lreg_neg: $lreg_neg/
        s/lrec_neg: .*/lrec_neg: $lrec_neg/
        s/scan_pretrain: .*/scan_pretrain: $scan_pretrain/
        }" config/ucf_test.yaml

    sed -i "/^ood:/,/^[^ ]/{
        s/lr: .*/lr: $lr_ood_ucf/
        s/bs: .*/bs: $batch_size_ood_ucf/
        s/syn_num: .*/syn_num: $syn_num_ood_ucf/
        }" config/ucf_test.yaml

    sed -i "s/seen_contrasive_ratio = .*/seen_contrasive_ratio = $seen_contrasive_ratio_ucf/" src/global_var.py
    python main.py config/ucf_test.yaml
    run_pid_ucf=$!
    sleep 5
fi

if [ $run_vgg = 1 ]; then
    if [ $seen_contrasive_ratio_vgg = 0 ]; then
        run_file_path=""$run_file_path_head"_"$r_enc_vgg"_"$r_proj_vgg"_"$r_dec_vgg""
    else
        run_file_path=""$run_file_path_head"_"$r_enc_vgg"_"$r_proj_vgg"_"$r_dec_vgg"_"$seen_contrasive_ratio_vgg""
    fi
    exp_name="vgg_"$exp_name_suffix""
    rm -r $run_file_path/$exp_name

    sed -i "/^[^ ]/,/^\s*$/{
        s/run_file_path: .*/run_file_path: $run_file_path/
        s/exp_name: .*/exp_name: $exp_name/
        }" config/vgg_test.yaml

    sed -i "/^dataset:/,/^[^ ]/{
        s/root_set: .*/root_set: $root_set/
        }" config/vgg_test.yaml

    sed -i "/^gen:/,/^[^ ]/{
        s/epochs: .*/epochs: $epochs_vgg/
        }" config/vgg_test.yaml

    sed -i "/^clss:/,/^[^ ]/{
        s/epochs: .*/epochs: 200/
        s/lr: .*/lr: $lr_clss_vgg/
        s/bs: .*/bs: $batch_size_clss_vgg/
        }" config/vgg_test.yaml

    sed -i "/^clsu:/,/^[^ ]/{
        s/epochs: .*/epochs: 50/
        s/num_workers: .*/num_workers: $num_workers/
        s/bs: .*/bs: $bs_vgg/
        s/n_batches: .*/n_batches: $n_batches_vgg/
        s/lr: .*/lr: $lr_vgg/
        s/r_enc: .*/r_enc: $r_enc_vgg/
        s/r_proj: .*/r_proj: $r_proj_vgg/
        s/r_dec: .*/r_dec: $r_dec_vgg/
        s/ltrip_neg: .*/ltrip_neg: $ltrip_neg/
        s/lreg_neg: .*/lreg_neg: $lreg_neg/
        s/lrec_neg: .*/lrec_neg: $lrec_neg/
        s/scan_pretrain: .*/scan_pretrain: $scan_pretrain/
        }" config/vgg_test.yaml

    sed -i "/^ood:/,/^[^ ]/{
        s/epochs: .*/epochs: 80/
        s/lr: .*/lr: $lr_ood_vgg/
        s/bs: .*/bs: $batch_size_ood_vgg/
        s/syn_num: .*/syn_num: $syn_num_ood_vgg/
        }" config/vgg_test.yaml

    sed -i "s/seen_contrasive_ratio = .*/seen_contrasive_ratio = $seen_contrasive_ratio_vgg/" src/global_var.py
    python main.py config/vgg_test.yaml
    run_pid_vgg=$!
    sleep 5
fi

