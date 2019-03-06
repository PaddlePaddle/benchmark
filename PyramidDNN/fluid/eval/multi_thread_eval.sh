if [ $# -ne 1 ];then
    echo "usage: $0 input_file"
	exit 1
fi

# resolve links - $0 may be a softlink
THIS="$0"
while [ -h "$THIS" ]; do
	ls=`ls -ld "$THIS"`
	link=`expr "$ls" : '.*-> \(.*\)$'`
	if expr "$link" : '.*/.*' > /dev/null; then
		THIS="$link"
	else
		THIS=`dirname "$THIS"`/"$link"
	fi
done

THIS_DIR=`dirname "$THIS"`

TMP_DIR="_tmp"`date +%F-%H-%M.%N`
mkdir -p $TMP_DIR

for i in {0..9}
do
    cat $1 | awk 'substr($1,length($1),1)==t{print$1"\t"$2"\t"int($3)}' t=$i | sh $THIS_DIR/count.sh > $TMP_DIR/tmp.eval_out.$i &
done
wait

cat $TMP_DIR/tmp.eval_out.* | awk '$1!="label"&&$1!="label_pair"{a[$1]+=$2} $1=="label"{label[$2]=1} $1=="label_pair"{label_pair[$2]=1}
    END{
	#for(x in a )print x,a[x]
	print "query number:\t"a["query_number"]"\tpair_number:\t"a["pair_number"]
	printf("pearson_2:\t%.4f\n",(a["zy"]-a["z"]/a["n"]*a["y"])/(sqrt(a["z2"]-a["z"]/a["n"]*a["z"])*sqrt(a["y2"]-a["y"]/a["n"]*a["y"])));

	printf("%s\t%.3f\t%.3f\t%d\t%d\t%d\n",  "=:=:", a["right"]/a["wrong"],(a["right"]+0.5*a["equal"])/(a["wrong"]+0.5*a["equal"]),a["right"],a["wrong"],a["equal"]);
	for(x in label)
	{
	    printf("%s\t%.3f\t%.3f\t%d\t%d\t%d\n",  "=:"x":", a["right_"x]/a["wrong_"x],(a["right_"x]+0.5*a["equal_"x])/(a["wrong_"x]+0.5*a["equal_"x]),a["right_"x],a["wrong_"x],a["equal_"x]);
	    printf("[%s]:\t%d\t%.4f\t",x,a["level_num_"x],a["level_sum_"x]/a["level_num_"x]);
	    printf("%.4f\n", sqrt(a["level_squared_sum_"x]/a["level_num_"x]-(a["level_sum_"x]/a["level_num_"x])*(a["level_sum_"x]/a["level_num_"x])));

	}
	for(x in label_pair)
	{
	    printf("%s\t%.3f\t%.3f\t%d\t%d\t%d\n",  x":", a["right_"x]/a["wrong_"x],(a["right_"x]+0.5*a["equal_"x])/(a["wrong_"x]+0.5*a["equal_"x]),a["right_"x],a["wrong_"x],a["equal_"x]);
	}
    }'| LC_ALL=C sort -t ':'  -n

#rm -rf $TMP_DIR
