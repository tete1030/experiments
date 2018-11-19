#!/bin/bash

if [[ -z $1 ]]; then
	echo "No dir"
	exit 1
fi

if [[ -z $2 ]]; then
	echo "No divider"
	exit 1
fi

if [[ ! $2 =~ [[:digit:]]+ ]]; then
	echo "Divider should be number"
	exit 1
fi

for file in "$1"/checkpoint_*.pth.tar ; do
	file_no=$(echo "$file" | sed -E 's/.*checkpoint_([[:digit:]]+)\.pth\.tar/\1/g') 
	if [[ $file_no =~ [[:digit:]]+ ]]; then
		if [[ $(( $file_no % $2 ))  == 0 ]]; then
			if [[ $3 != "yes" ]]; then
				echo "### $file"
			fi
		else
			if [[ $3 == "yes" ]]; then
				rm "$file"
			else
				echo "!!! $file"
			fi
		fi
	fi
done
