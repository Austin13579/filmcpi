for runseed in 0 1 2 3 4
do
python split.py --ds biosnap --rs $runseed --split up
python split.py --ds biosnap --rs $runseed --split ud
python split.py --ds biosnap --rs $runseed --split random
done

