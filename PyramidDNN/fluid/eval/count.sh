awk -F"\t" '
function f()
{
    for(i=1;i<un;i++)
    {
	for(j=i+1;j<=un;j++)
	{
	    if(label[i]==label[j])
		continue
	    large=i;
	    small=j;
	    if(label[i]<label[j])
	    {
		large=j;
		small=i;
	    }
	    s=label[large];
	    s_detail=label[large]":"label[small];
	    if(score[large]>score[small])
	    {
		right++;
		r[s]++;
		r_d[s_detail]++;
	    }
	    else if(score[large]<score[small])
	    {
		wrong++;
		w[s]++;
		w_d[s_detail]++;
	    }
	    else
	    {
		equal++;
		e[s]++;
		e_d[s_detail]++;
	    }
	}
    }
}
BEGIN{
    un=0;
    preq="";
    debug_n=0;
    scale[0]=1;scale[1]=2;scale[2]=4;scale[3]=8;scale[4]=16;
}
{
    if(NR>1 && $1!=preq)
     {
	 f();
	 debug_n+=un;
	 un=0;
	 qnum++;
     }
    un++;
    preq=$1;
    score[un]=$2;
    label[un]=$3;
    
    level_num[$3]++;
    level_sum[$3]+=$2
    level_squared_sum[$3]+=$2*$2
    
    z+=$2;
    z2+=$2*$2;
    y+=scale[$3];
    y2+=scale[$3]*scale[$3];
    zy+=$2*scale[$3]; 
    
    n++;
}
END{
    f();
    print "query_number\t"(qnum+1)"\npair_number\t"(right+equal+wrong)
    
    for(x in level_num)
    {
	print "level_num_"x"\t"level_num[x];
	print "level_sum_"x"\t"level_sum[x];
	print "level_squared_sum_"x"\t"level_squared_sum[x]
    }
    
    print "z\t"z"\ny\t"y"\nz2\t"z2"\ny2\t"y2"\nzy\t"zy"\nn\t"n;
    
    print "right\t"right
    print "wrong\t"wrong
    print "equal\t"equal
    for( x in w)
    {
	print "right_"x"\t"r[x]
	print "wrong_"x"\t"w[x]
	print "equal_"x"\t"e[x]
	print "label\t"x
    }
    for( x in w_d)
    {	print "right_"x"\t"r_d[x]
	print "wrong_"x"\t"w_d[x]
	print "equal_"x"\t"e_d[x]
	print "label_pair\t"x
    }
}
'
