# !/bin/bash
size=12661
folder="../data/random"
learn="learn"
test="test"
n=$1
m=$((size - n))

cd $folder
for i in `ls r*`;
do
    echo "on $i"
    head -n 1 $i > "${test}_$i"
    tail -n $n $i >> "${test}_$i"
    head -n $m $i > "${learn}_$i"
    echo "$i finished"
done
	 
