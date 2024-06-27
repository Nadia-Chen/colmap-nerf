# colmap-nerf

训练和测试指令：（边训练边测试）
python train.py --root_dir cus_data --exp_name 626 --eval_lpips --scale 4 --num_epochs 30
--root_dir：root directory of dataset
--scale：scene scale (whole scene must lie in [-scale, scale]^3

展示gui指令：
python show_gui.py --root_dir cus_data --ckpt_path ckpts/colmap/626_4/epoch=29_slim.ckpt --scale 4 
--ckpt_path：<path/to/.ckpt>

其他具体参数请详细看opt.py文件
