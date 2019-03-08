其实就是计算 每个query 所有 正/负title组成的pair中，正序pair与逆序pair数量的比： count(pos_score > neg_score) / count(pos_score < neg_score) 。

输入是这样的格式： “\t”切分， $1==qid, $2==score, $3==label。 脚本会自动去在同一个qid下组pair
