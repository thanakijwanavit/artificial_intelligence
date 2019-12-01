start_time=`date +%s`
number_of_run=$1
python train.py -r $number_of_run&& echo run time is $(expr $(expr `date +%s` - $start_time) / $number_of_run) s per round, total time is $(expr `date +%s` - $start_time) s
#python merge_q.py