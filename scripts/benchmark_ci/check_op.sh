for list in `cat operatorlist`
do
#extract op name from list
tmp=${list##*/}
op=${tmp%.*}
echo "{op}"
done
