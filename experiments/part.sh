ID=0
ARCH=hg
STACK=3
BLOCK=1
EPOCHS=220
TRAINBATCH=12
TESTBATCH=12
WORKER=16
LR=5e-4
# RESUME="--resume checkpoint/mpii/${ARCH}_part_S${STACK}_B${BLOCK}/checkpoint.pth.tar"
# EVALUATE="--evaluate"

cd ~/my/pytorch-pose

CUDA_VISIBLE_DEVICES=${ID} python experiments/hg_part.py \
   -a ${ARCH} \
   -s ${STACK} --blocks ${BLOCK} \
   --checkpoint checkpoint/mpii/${ARCH}_part_S${STACK}_B${BLOCK} \
   --epochs ${EPOCHS} \
   --train-batch ${TRAINBATCH} \
   --test-batch ${TESTBATCH} \
   -j ${WORKER} \
   --lr ${LR} \
   --schedule 150 175 200 \
   ${RESUME} \
   ${EVALUATE}

