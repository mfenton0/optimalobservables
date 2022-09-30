n
generate p p > t t~
#define l = e+ mu+ e- mu-
#define v = ve vm ve~ vm~
launch
madspin=ON
shower=Pythia8
detector=Delphes
analysis=OFF
decay t > w+ b, w+ > l+ vl
decay w+ > l+ vl
decay t~ > w- b~, w- > l- vl~
decay w- > l- vl~
set nevents 40000
#set spinmode none
#launch
