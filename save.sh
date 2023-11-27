#! /bin/bash

# Disable a specific shellcheck warning (SC2162: read without -r will mangle backslashes)
# shellcheck disable=SC2162

# Read each line from the file 'matrices.txt'
while read input; do
  # Pass each line (content of 'input') as input to the './main' program
  # and redirect the output of './main' to a file named after the input line with '.out' extension
  echo "$input" | ./main > "$input.out"
done < matrices.txt
