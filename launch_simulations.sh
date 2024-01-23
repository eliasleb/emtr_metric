#!/usr/bin/env zsh

# args = [(WindowedSineExcitation(fi, 2), 2.5 / fi, 3 / fi, 0 / fi, 4, res) for fi in fs] + \
#        [(WindowedSineExcitation(fi, 4), 4.5 / fi, 5 / fi, 0 / fi, 5.5, res) for fi in fs] + \
#        [(AsymmetricExcitation(fi / 2, fi, 1 / fi), 6 / fi, 6 / fi, 0., 6., res) for fi in fs] + \
#        [(DerivedGaussianExcitation(fi), 2 / fi, 2 / fi, 0., 2, res) for fi in fs]

(( res = 1 ))

for f in $(seq 0.5 0.1 2)
do
  command1="""python main.py WindowedSineExcitation "$(bc -l <<< "2.5/$f")" "$(bc -l <<< "3/$f")" 4. $res $f 2"""
  command2="""python main.py WindowedSineExcitation "$(bc -l <<< "4.5/$f")" "$(bc -l <<< "5/$f")" 5.5 $res $f 4"""
  command3="""python main.py AsymmetricExcitation "$(bc -l <<< "6./$f")" "$(bc -l <<< "6./$f")" 6. $res "$(bc -l <<< "$f/2")" $f "$(bc -l <<< "1/$f")" """
  command4="""python main.py DerivedGaussianExcitation "$(bc -l <<< "2/$f")" "$(bc -l <<< "2/$f")" 2. $res $f"""
  eval ${command1}
  eval ${command2}
  eval ${command3}
  eval ${command4}

done
