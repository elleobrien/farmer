#!/usr/bin/env bash
all=();
for i in $(git ls-remote --heads https://github.com/ShrutiC-git/farmer.git);
do
all+=($i);
done

length=${#all[@]}

sha=()
for ((i=0; i<$length; i++));
do
if (($i%2==0));
then sha+=(${all[$i]});
fi;
done

#for i in ${sha[@]}
#do echo $i
#done
echo "::set-output name=TAG_NAME"::$sha
