#!/bin/bash
current_dir=$(pwd)
for dir in "$current_dir"/*/; do
  # Check if the directory is not "include"
  if [ "$dir" != "$current_dir/include/" ]; then
    # Change to the directory
    cd "$dir"

    # Run the clear.sh script
    chmod +x clean.sh
    ./clean.sh
    # Run the cc.sh script
    chmod +x cc.sh
    ./cc.sh

    # Run the run.sh script
    chmod +x run.sh
    ./run.sh
    ./clean.sh

    # Change back to the original directory
    cd -
  fi
done
