#! /bin/bash
CURRENT_DIR=$(pwd)

FILE_LIST=$CURRENT_DIR/corpus/MAPS/list/$2.list
FILE_CONFIG=$CURRENT_DIR/corpus/MAPS/dataset/config.json
DIR_FEATURE=$CURRENT_DIR/corpus/MAPS/feature
DIR_REFERENCE=$CURRENT_DIR/corpus/MAPS/reference

DIR_CHECKPOINT=$CURRENT_DIR/checkpoint/MAPS/1FLT

DIR_RESULT=$CURRENT_DIR/result/MAPS/1FLT
mkdir -p $DIR_RESULT

# 2FDT/1FLT
MODE=combination
OUTPUT=2nd
# 1FDN
#MODE=single
#OUTPUT=1st

# inference
python3 $CURRENT_DIR/evaluation/m_inference.py -f_list $FILE_LIST -f_config $FILE_CONFIG -d_cp $DIR_CHECKPOINT -m $1 -d_fe $DIR_FEATURE -d_mpe $DIR_RESULT -d_note $DIR_RESULT -calc_transcript -mode $MODE -ablation

# mir_eval
python3 $CURRENT_DIR/evaluation/m_transcription.py -f_list $FILE_LIST -d_ref $DIR_REFERENCE -d_est $DIR_RESULT -d_out $DIR_RESULT -output $OUTPUT
python3 $CURRENT_DIR/evaluation/m_transcription.py -f_list $FILE_LIST -d_ref $DIR_REFERENCE -d_est $DIR_RESULT -d_out $DIR_RESULT -output $OUTPUT -velocity
python3 $CURRENT_DIR/evaluation/m_mpe.py -f_config $FILE_CONFIG -f_list $FILE_LIST -d_ref $DIR_REFERENCE -d_est $DIR_RESULT -d_out $DIR_RESULT -output $OUTPUT -thred_mpe 0.5
