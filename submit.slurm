#!/bin/bash 
#MODES=${@} 

START=$1
END=$2

#for mode in ${MODES} ;
 for i in {${START}..${END}}
  do 
     sbatch -n 1 -p atlas -t 10:00:00 --mem-per-cpu=2G generate.sh $i SPIN
     sbatch -n 1 -p atlas -t 10:00:00 --mem-per-cpu=2G generate.sh $i NOSPIN

  done 
