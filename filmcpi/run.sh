for runseed in 0 1 2 3 4
do
python -u train.py --rs $runseed --ds biosnap --split up --model_type film_cp
python -u train.py --rs $runseed --ds biosnap --split ud --model_type film_cp
python -u train.py --rs $runseed --ds biosnap --split random --model_type film_cp
done
