GPUID=0
ARCH=hg
STACK=3
BLOCK=1
EPOCHS=220
TRAINBATCH=12
TESTBATCH=12
WORKER=24
LR=5e-4
EXP="part"
AFFIX="3"
EXPERIMENT_NAME="${ARCH}_${EXP}_S${STACK}_B${BLOCK}_${AFFIX}"
CHECKPOINT_PATH="checkpoint/mpii/${EXPERIMENT_NAME}"
# RESUME="--resume ${CHECKPOINT_PATH}/checkpoint.pth.tar"
# EVALUATE="--evaluate"

cd ../

CUDA_VISIBLE_DEVICES=${GPUID} python experiments/hg_part.py \
   --exp ${EXP} \
   -a ${ARCH} \
   -s ${STACK} --blocks ${BLOCK} \
   --checkpoint "${CHECKPOINT_PATH}" \
   --epochs ${EPOCHS} \
   --train-batch ${TRAINBATCH} \
   --test-batch ${TESTBATCH} \
   -j ${WORKER} \
   --lr ${LR} \
   --schedule 150 175 200 \
   --selective "experiments/sel.npy" \
   --skip-val 5 \
   --hyperdash "HG-Part" \
   ${RESUME} \
   ${EVALUATE}

