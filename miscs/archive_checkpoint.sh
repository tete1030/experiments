EXP_ID=$1
DIR=$2
COMPARE=$3

mkdir checkpoint/${EXP_ID}/${DIR} && find checkpoint/${EXP_ID}/ -maxdepth 1 -type f -newer checkpoint/${EXP_ID}/${COMPARE} -exec mv {} checkpoint/${EXP_ID}/${DIR}/ \;
