# Temp (no save)
# python main.py --model slist --dataset diginetica --train_weight 1.0 --alpha 0.9 --predict_weight 2.0  --folder SLIST_final
# python main.py --model slist --dataset retailrocket --train_weight 0.5 --alpha 0.9 --predict_weight 4.0  --folder SLIST_final
# python main.py --model slist --dataset tmall --train_weight 1.0 --alpha 0.9 --predict_weight 4.0  --folder SLIST_final
# python main.py --model slist --dataset dressipi --train_weight 1.0 --alpha 0.7 --predict_weight 4.0  --folder SLIST_final
# python main.py --model slist --dataset lastfm --train_weight 1.0 --alpha 0.5 --predict_weight 4.0  --folder SLIST_final

# Temp (no save)
# python main.py --model swalk --dataset diginetica --predict_weight 2.0 --save_load_SLIST False --model_transition SLIT --model_teleportation SLIS --walk_p 0.5 --self_beta 0.7 --folder SWalk_final
# python main.py --model swalk --dataset yoochoose --predict_weight 1.0 --save_load_SLIST False --model_transition SLIT --model_teleportation SLIS --walk_p 0.5 --self_beta 1.0 --folder SWalk_final --save_path saved/yoochoose_SWALK
# python main.py --model swalk --dataset retailrocket --predict_weight 4.0 --save_load_SLIST False --model_transition SLIT --model_teleportation SLIS --walk_p 0.5 --self_beta 0.7 --folder SWalk_final --save_path saved/retailrocket_SWALK
# python main.py --model swalk --dataset tmall --predict_weight 4.0 --save_load_SLIST False --model_transition SLIT --model_teleportation SLIS --walk_p 0.1 --self_beta 0.1 --folder SWalk_final
# python main.py --model swalk --dataset dressipi --predict_weight 4.0 --save_load_SLIST False --model_transition SLIT --model_teleportation SLIS --walk_p 0.5 --self_beta 1.0 --folder SWalk_final
# python main.py --model swalk --dataset lastfm --predict_weight 4.0 --save_load_SLIST False --model_transition SLIT --model_teleportation SLIS --walk_p 0.5 --self_beta 1.0 --folder SWalk_final

# LINK
TEACHER=core_trm
python main.py --model link --dataset diginetica --predict_weight 2.0 --reg_teacher 100 --agg_method closed_form --slis_alpha 0.9 --teacher_path saved_models_for_embedding/linear_teacher_diginetica_${TEACHER}/dense_matrix.npy --folder link_final --teacher_normalize True --teacher_temperature 0.1
python main.py --model link --dataset yoochoose --predict_weight 1.0 --reg_teacher 1000 --agg_method closed_form --slis_alpha 0.5 --teacher_path saved_models_for_embedding/linear_teacher_yoochoose_${TEACHER}/dense_matrix.npy --folder link_final --teacher_normalize True --teacher_temperature 0.1
python main.py --model link --dataset retailrocket --predict_weight 4.0 --reg_teacher 100 --agg_method closed_form --slis_alpha 0.9 --teacher_path saved_models_for_embedding/linear_teacher_retailrocket_${TEACHER}/dense_matrix.npy --folder link_final --teacher_normalize True --teacher_temperature 0.1
python main.py --model link --dataset tmall --predict_weight 4.0 --reg_teacher 1000 --agg_method closed_form --slis_alpha 0.9 --teacher_path saved_models_for_embedding/linear_teacher_tmall_${TEACHER}/dense_matrix.npy --folder link_final --teacher_normalize True --teacher_temperature 0.05
python main.py --model link --dataset dressipi --predict_weight 4.0 --reg_teacher 100 --agg_method closed_form --slis_alpha 0.9 --teacher_path saved_models_for_embedding/linear_teacher_dressipi_${TEACHER}/dense_matrix.npy --folder link_final --teacher_normalize True --teacher_temperature 0.1
python main.py --model link --dataset lastfm --predict_weight 8.0 --reg_teacher 100 --agg_method closed_form --slis_alpha 0.9 --teacher_path saved_models_for_embedding/linear_teacher_lastfm_${TEACHER}/dense_matrix.npy --folder link_final --teacher_normalize True --teacher_temperature 0.1  