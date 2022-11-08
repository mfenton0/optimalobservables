#!/bin/bash

cd /DFS-L/DATA/atlas/mjfenton/.tmp_batch
source /export/nfs0home/mjfenton/.bashrc
#ml miniconda/3/own 
#conda init bash
conda activate mg5_aMC
source /DFS-L/DATA/atlas/mjfenton/MG5_aMC_v3_4_1/root/bin/thisroot.sh

pwd
ls -ltrh

OUTDIR=/DFS-L/DATA/atlas/mjfenton/optimalobservables/reconstructed_events/
MGDIR=/DFS-L/DATA/atlas/mjfenton//MG5_aMC_v3_4_1/
SEED=$1
SPINMODE=$2

FILEDIR="./output_${SPINMODE}_${SEED}"
RAND=${RANDOM}
TMPDIR="mjfenton${SEED}${SPINMODE}${RAND}"

mkdir -p ${TMPDIR}
cd ${TMPDIR}

pwd

cp /DFS-L/DATA/atlas/mjfenton/optimalobservables/*.py .
cp /DFS-L/DATA/atlas/mjfenton/optimalobservables/mg5.sh .

ls -ltrh

if [ "${SPINMODE}" = "NOSPIN" ]; then
    echo "Generating samples with no spin correlation"
    cp /DFS-L/DATA/atlas/mjfenton/optimalobservables/mg5_nospin.sh ./mg5.sh
fi



#conda activate mg5_aMC
${MGDIR}/bin/mg5_aMC mg5.sh
#conda deactivate

echo "random_seed = $1" > reco_config.py
echo 'root_file_path = (' >> reco_config.py
echo '    "./PROC_sm_0/"' >> reco_config.py
echo '    "Events/run_01_decayed_1/tag_1_delphes_events.root"' >> reco_config.py
echo ")" >> reco_config.py
echo "recos_output_dir = \"${FILEDIR}\"" >> reco_config.py

cat reco_config.py

#sed -i "s/=\ 0/=\ $1/g" reco_config.py

python reconstruct.py

mv ${FILEDIR} ${OUTDIR}
cd ../
rm -r ${TMPDIR}
