#! /bin/bash

while read input; do
  echo "$input" | ./main > "$input.out"
done < matrices.txt
