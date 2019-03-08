from __future__ import print_function

import paddle
import numpy as np
import six

def train_file_read(data_file='./train_data_6'):
    max_seq_len = 1024

    def convert(s):
        # if we meets empty sentence, replace it with an 0L to align with Lego
        if s == "":
            return [0L]

        if six.PY2:
            vector_list = map(lambda x: long(x), s.split(' '))
        else:
            vector_list = map(lambda x: int(x), s.split(' '))
        return vector_list if len(vector_list) <= max_seq_len else vector_list[0:1024]

    def reader():
        with open(data_file) as f:
            #  i = 0
            #  sum_len = 0
            for line in f.readlines():
                #  i += 1
                pairs = line.split(';')
                
                pt_num, nt_num = convert(pairs[1])

                assert len(pairs) == (pt_num + nt_num) * 2 + 5
                query_basic = convert(pairs[2])
                query_phrase = convert(pairs[3])
                pos_basic_titles = []
                pos_phrase_titles = []
                neg_basic_titles = []
                neg_phrase_titles = []
                for i in range(pt_num):
                    pos_basic_titles.append(convert(pairs[4 + 2*i]))
                    pos_phrase_titles.append(convert(pairs[4 + 2*i + 1]))
	        for j in range(nt_num):
		    neg_basic_titles.append(convert(pairs[4 + 2*pt_num + 2*j]))
		    neg_phrase_titles.append(convert(pairs[4 + 2*pt_num + 2*j + 1]))
                sampler = ConstantSampler()    
                for pos_basic, pos_phrase, neg_basic, neg_phrase, label in sampler.sample(pos_basic_titles, pos_phrase_titles, neg_basic_titles, neg_phrase_titles):
		    yield query_basic, query_phrase, pos_basic, pos_phrase, neg_basic, neg_phrase, label

                #  if i % 128 == 0:
                    #  sum_len = sum_len + len(query_basic) + len(query_phrase) + len(pos_title_basic) + len(pos_title_phrase) + len(neg_title_basic) + len(neg_title_phrase) + len(label)
                    #  print(i, sum_len)
                    #  import sys
                    #  sys.stdout.flush()
                    #  sum_len = 0
                #  else:
                    #  sum_len = sum_len + len(query_basic) + len(query_phrase) + len(pos_title_basic) + len(pos_title_phrase) + len(neg_title_basic) + len(neg_title_phrase) + len(label)

                #yield query_basic, query_phrase, pos_title_basic, pos_title_phrase, neg_title_basic, neg_title_phrase, label
            #  #  while True:
                #  #  yield [1] * 100, [2] * 100, [3] * 100, [4] * 100, [5] * 100, [6] * 1024, [0]
                #  #  yield [1], [2], [3], [4], [5], [6], [0]

    #  def reader():
        #  with open(data_file) as f:
            #  import mmap
            #  m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            #  line = m.readline()
            #  while line:
                #  pairs = line.split(';')

                #  assert len(pairs) == 7

                #  query_basic = convert(pairs[0])
                #  query_phrase = convert(pairs[1])
                #  pos_title_basic = convert(pairs[2])
                #  pos_title_phrase = convert(pairs[3])
                #  neg_title_basic = convert(pairs[4])
                #  neg_title_phrase = convert(pairs[5])
                #  label = convert(pairs[6])

                #  yield query_basic, query_phrase, pos_title_basic, pos_title_phrase, neg_title_basic, neg_title_phrase, label
                #  line = m.readline()
            #  #  while True:
                #  #  yield [1] * 100, [2] * 100, [3] * 100, [4] * 100, [5] * 100, [6] * 1024, [0]
                #  #  yield [1], [2], [3], [4], [5], [6], [0]

    return reader

def test(test_file='./eval_data_5'):
    def convert(s):
        # if we meets empty sentence, replace it with an 0L to align with Lego
        if s == "":
            return [0L]

        if six.PY2:
            vector_list = map(lambda x: long(x), s.split(' '))
        else:
            vector_list = map(lambda x: int(x), s.split(' '))
        return vector_list

    def reader():
        with open(test_file) as f:
            for line in  f.readlines():
                pairs = line.split(';')

                assert len(pairs) == 6

                line_count = int(pairs[0].split(' ')[0])
                query_basic = convert(pairs[1])
                query_phrase = convert(pairs[2])
                title_basic = convert(pairs[3])
                title_phrase = convert(pairs[4])
                label = convert(pairs[0].split(' ')[1])

                yield line_count, query_basic, query_phrase, title_basic, title_phrase, label
                #  yield query_basic, query_phrase, title_basic, title_basic

    return reader

def convert_from_lego_to_fluid(src_file, train_file, eval_file, sampler, is_test=False):
    with open(src_file, 'r') as src_f:
        with open(train_file, 'a+') as dst_f:
            with open(eval_file, 'a+') as eva_f:
                line_count = 1
                for line in src_f.readlines():
                    words = line.split(';')

                    title_num = words[1].split(' ')
                    assert len(title_num) == 2
                    num_of_pos_titles = int(title_num[0])
                    num_of_neg_titles = int(title_num[1])

                    query_basic = words[2]
                    query_phrase = words[3]

                    pos_start = 4
                    pos_end = 4 + num_of_pos_titles * 2
                    neg_start = pos_end
                    neg_end = pos_end + num_of_neg_titles * 2

                    pos_basic_titles = []
                    pos_phrase_titles = []
                    pos_titles = words[pos_start:pos_end]
                    count = 0
                    for pos in pos_titles:
                        if count % 2 == 0:
                            pos_basic_titles.append(pos)
                        else:
                            pos_phrase_titles.append(pos)
                        count += 1
                    assert len(pos_basic_titles) == len(pos_phrase_titles)

                    neg_basic_titles = []
                    neg_phrase_titles = []
                    neg_titles = words[neg_start:neg_end]
                    count = 0
                    for neg in neg_titles:
                        if count % 2 == 0:
                            neg_basic_titles.append(neg)
                        else:
                            neg_phrase_titles.append(neg)
                        count += 1
                    assert len(neg_basic_titles) == len(neg_phrase_titles)

                    if not is_test:
                    #  if line_count > 500:
                        for pos_basic, pos_phrase, neg_basic, neg_phrase, label in sampler.sample(pos_basic_titles, pos_phrase_titles, neg_basic_titles, neg_phrase_titles):
                            sample = ";".join([query_basic, query_phrase, pos_basic, pos_phrase, neg_basic, neg_phrase, label])
                            dst_f.write(sample + '\n')
                    else:
                        for pos_basic, pos_phrase in six.moves.zip(pos_basic_titles, pos_phrase_titles):
                            sample = ";".join([str(line_count), query_basic, query_phrase, pos_basic, pos_phrase, "1"])
                            eva_f.write(sample + '\n')
                        for neg_basic, neg_phrase in six.moves.zip(neg_basic_titles, neg_phrase_titles):
                            sample = ";".join([str(line_count), query_basic, query_phrase, neg_basic, neg_phrase, "0"])
                            eva_f.write(sample + '\n')
                    line_count += 1



class ConstantSampler(object):
    def __init__(self):
        pass

    def sample(self, pos_basic_titles, pos_phrase_titles, neg_basic_titles, neg_phrase_titles):
        #np.random.seed(1)
        for neg_basic, neg_phrase in six.moves.zip(neg_basic_titles, neg_phrase_titles):
            idx = np.random.randint(0, len(pos_basic_titles))
            yield pos_basic_titles[idx], pos_phrase_titles[idx], neg_basic, neg_phrase, '1'


#  train_reader = train("./train_data_6")

test_reader = test("./eval_data_6")

if __name__ == "__main__":
    #  convert_from_lego_to_fluid('./test.out.traindata_pyramid_20180204_ids_wise_1month', './train_data_3', './eval_data_3', ConstantSampler())
    convert_from_lego_to_fluid('./mini_data.origin', './train_data_5', './eval_data_5', ConstantSampler())

    #convert_from_lego_to_fluid('./mini_data_test.origin', './train_data_5', './eval_data_5', ConstantSampler(), is_test=True)

    ##convert_from_lego_to_fluid('./test.out', './train_data_little', './eval_data_little', ConstantSampler())
    #  reader = paddle.batch(train("./train_data_5"), batch_size=1)
    #  count = 0
    #  with open('yyy', 'w') as f:
        #  for data in reader():
            #  data = data[0]
            #  count += 1
            #  if count > 500:
                #  break
            #  #  print(' '.join([ str(x) for x in data[1] ]))

            #  x = ';'.join((' '.join([ str(x) for x in data[0] ]),
                          #  ' '.join([ str(x) for x in data[1] ]),
                          #  ' '.join([ str(x) for x in data[2] ]),
                          #  ' '.join([ str(x) for x in data[3] ]),
                          #  ' '.join([ str(x) for x in data[4] ]),
                          #  ' '.join([ str(x) for x in data[5] ])))
            #  f.write(x + ';' + str(data[6][0]) + '\n')
        #  f.flush()

    #  reader = test("./eval_data_5")
    #  count = 0
    #  with open('xxx', 'w') as f:
        #  for data in reader():
            #  #  print(' '.join([ str(x) for x in data[1] ]))

            #  x = ';'.join((' '.join([ str(x) for x in data[1] ]),
                          #  ' '.join([ str(x) for x in data[2] ]),
                          #  ' '.join([ str(x) for x in data[3] ]),
                          #  ' '.join([ str(x) for x in data[4] ])))
            #  f.write(str(data[0]) + ';' + x + ';' + str(data[7][0]) + '\n')
        #  f.flush()
