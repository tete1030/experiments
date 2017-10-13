ID=0
ARCH=hg
STACK=8
BLOCK=1
EPOCHS=220
TRAINBATCH=12
TESTBATCH=12
WORKER=20
LR=5e-4
RESUME="--resume checkpoint/mpii/model_best.pth.tar"
EVALUATE="--evaluate"

cd ~/my/pytorch-pose

CUDA_VISIBLE_DEVICES=${ID} python example/mpii.py \
   -a ${ARCH} \
   -s ${STACK} --blocks ${BLOCK} \
   --checkpoint checkpoint/mpii/${ARCH}_S${STACK}_B${BLOCK} \
   --epochs ${EPOCHS} \
   --train-batch ${TRAINBATCH} \
   --test-batch ${TESTBATCH} \
   -j ${WORKER} \
   --lr ${LR} \
   --schedule 150 175 200 \
   ${RESUME} \
   ${EVALUATE}

cd -
