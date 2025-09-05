#!/bin/bash --login

PROJECT_NAME=fixed_eps_Ttf
SCRIPTDIR="./src/scripts"
BINDIR="./src/bin"
LOGDIR="./data_assets/logs"
MODELDIR="./data_assets/models/"

ACTF=tanh
R=2
K=16
M_mu=0
M_sig=1
W_mu=0
W_sig=1
Z_mu=0
Z_sig=1
BS=4

# Original parameters
# NSEED=10
# EPOCHS=100000

# For testing
NSEED=2
EPOCHS=10

mkdir -p ${LOGDIR}
mkdir -p ${MODELDIR}/${PROJECT_NAME}/

INPUT_PARAMS="n_seed = ${NSEED} \n r = ${R} \n K = ${K} \n bs = ${BS} \n epochs = ${EPOCHS} \n actf = ${ACTF}"

echo ${INPUT_PARAMS} > ${MODELDIR}/${PROJECT_NAME}/README.txt

# LR, BS
# Original parameters
# 144 tasks
#declare -a LRS=( 256.0 128.0 64.0 32.0 16.0 8.0 4.0 2.0 1.0 0.5 0.25 0.125 )
#declare -a EPSS=( 256.0 128.0 64.0 32.0 16.0 8.0 4.0 2.0 1.0 0.5 0.25 0.125 )

# For testing
declare -a LRS=( 1.0 2.0 )
declare -a EPSS=( 1.0 2.0 )

for i in "${LRS[@]}"
do
  for j in "${EPSS[@]}"
  do
    MODEL_NAME=r${R}K${K}_${ACTF}_lr${i}_bs${BS}_eps${j}
    LOG_FILE=${LOGDIR}/${MODEL_NAME}.log
    ERR_FILE=${LOGDIR}/${MODEL_NAME}.err
    nohup python $BINDIR/generate.py --actf ${ACTF} --r ${R} --K ${K} --M_mu ${M_mu} --M_sig ${M_sig} --W_mu ${W_mu} --W_sig ${W_sig} --Z_mu ${Z_mu} --Z_sig ${Z_sig} --bs ${BS} --lr ${i} --n_seed ${NSEED} --Wp_sig ${j} --epochs ${EPOCHS} --project ${PROJECT_NAME} --model_dir ${MODELDIR} > ${LOG_FILE} 2> ${ERR_FILE} &
  done
done
wait
