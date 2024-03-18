# modified classifier_sample.sh
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
LABEL_FLAGS="--labels 1"
RONAN_FLAGS="--input_selection_name /root/ronan-adm/1_adm_153.PNG --num_iter 2 --strategy min --lr 0.1"
python classifier_sample_recon.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS $LABEL_FLAGS $RONAN_FLAGS